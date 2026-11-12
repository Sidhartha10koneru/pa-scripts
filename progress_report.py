"""
progress_report.py - one ruler for the whole story: benchmark -> quality codes excluded -> passenger relabel.

Every model is scored on the SAME services (frozen official test set, quality codes left out, present in every run)
under TWO answer keys:
  original   GROUP_LABEL (frozen test set)                        - what the benchmark was measured on
  corrected  passenger-fix labels (y of the l6pax run's test set)   - passengers counted as not-PA
and at two levels: service (CSN + CPT, the prediction unit) and visit (CSN; PA if any service PA; score = max).

For every model x key x level:
  AUC-PR, AUC-ROC
  at a fixed catch rate (recall 75% and 80%): score cut-off, TP, FP, FN, TN, precision, services flagged
  at the fixed score cut-off 0.8:             recall, TP, FP, FN, TN, precision
Also: data funnel + train/validation/test counts of the final run, and precision-recall curves (PNG + CSV).

Run (edit the folder names if yours differ):
  python progress_report.py --frozen frozen/test_v1_2026-05_2026-07 \
     --runs bench_s42=exp_nofilter_2022 bench_s7=exp_keep_s7 \
            excl_s42=exp_qc_exclude_rv excl_s7=exp_exclude_s7 \
            pax_s42=exp_v8d_l6pax_s42 pax_s7=exp_v8d_l6pax_s7 \
     --corrected-from exp_v8d_l6pax_s42 --out progress 2>&1 | tee run_progress.log
Outputs: progress/metrics.csv, progress/summary.md, progress/pr_curves.csv, progress/pr_service_corrected.png
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

QUALITY = r"^\d{4}F$"


def norm(t):
    t["CSN"] = t.CSN.astype(str).str.replace(r"\.0$", "", regex=True)
    t["FINAL_CPT"] = t.FINAL_CPT.astype(str).str.strip().str.upper()
    return t


def run_services(folder):
    t = pd.read_parquet(f"{folder}/test_preds.parquet")
    t = norm(t.loc[:, ~t.columns.duplicated()])
    p = [c for c in t.columns if c.startswith("p_")]
    if len(p) != 1:
        raise SystemExit(f"{folder}: expected one score column p_*, found {p}")
    t = t[~t.FINAL_CPT.str.fullmatch(QUALITY)]
    return t.groupby(["CSN", "FINAL_CPT"], observed=True).agg(y=("y", "max"), s=(p[0], "max"))


def confusion(y, f):
    tp, fp = int((f & y).sum()), int((f & ~y).sum())
    fn, tn = int((~f & y).sum()), int((~f & ~y).sum())
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "flagged": tp + fp,
            "recall": round(tp / max(tp + fn, 1), 4), "precision": round(tp / max(tp + fp, 1), 4)}


def evaluate(y, s):
    y, s = np.asarray(y).astype(bool), np.asarray(s, dtype=float)
    out = [{"point": "overall", "AUC_PR": round(average_precision_score(y, s), 4), "AUC_ROC": round(roc_auc_score(y, s), 4),
            "n": len(y), "PA": int(y.sum())}]
    for rec in (0.75, 0.80):
        cut = float(np.quantile(s[y], 1 - rec))
        out.append({"point": f"recall {int(rec * 100)}%", "cutoff": round(cut, 4), **confusion(y, s >= cut)})
    out.append({"point": "cut-off 0.8", "cutoff": 0.8, **confusion(y, s >= 0.8)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="name=folder; names like bench_s42, excl_s42, pax_s42")
    ap.add_argument("--corrected-from", required=True, help="run folder whose test labels are the passenger-fix labels")
    ap.add_argument("--out", default="progress")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    runs = dict(r.split("=", 1) for r in a.runs)

    # ---- the ruler: frozen official services x every run x both answer keys
    fz = norm(pd.read_parquet(f"{a.frozen}/test_set_services.parquet"))
    S = fz[fz.IS_QUALITY_CODE == 0].set_index(["CSN", "FINAL_CPT"])[["y"]].rename(columns={"y": "y_original"})
    print(f"frozen official services: {len(S):,} | PA {int(S.y_original.sum()):,}")
    corr = run_services(a.corrected_from)[["y"]].rename(columns={"y": "y_corrected"})
    S = S.join(corr, how="inner")
    for n, f in runs.items():
        sv = run_services(f)
        S = S.join(sv[["s"]].rename(columns={"s": f"s_{n}"}), how="inner")
        print(f"  {n:10s} {f}: scored services {len(sv):,}")
    print(f"common services (every run + both keys): {len(S):,} | PA original {int(S.y_original.sum()):,} "
          f"| PA corrected {int(S.y_corrected.sum()):,} | disagree {int((S.y_original != S.y_corrected).sum()):,}")
    V = S.groupby(level="CSN").max()
    print(f"visits: {len(V):,} | PA original {int(V.y_original.sum()):,} | PA corrected {int(V.y_corrected.sum()):,}")

    rows, curves = [], []
    for level, D in [("service", S), ("visit", V)]:
        for key in ("original", "corrected"):
            for n in runs:
                for r in evaluate(D[f"y_{key}"], D[f"s_{n}"]):
                    rows.append({"level": level, "labels": key, "model": n, **r})
                if level == "service" and key == "corrected":
                    p, rc, _ = precision_recall_curve(D[f"y_{key}"].astype(bool), D[f"s_{n}"])
                    idx = np.unique(np.searchsorted(-rc[::-1], -np.linspace(1, 0, 201))).clip(0, len(rc) - 1)
                    curves.append(pd.DataFrame({"model": n, "recall": rc[::-1][idx], "precision": p[::-1][idx]}))
    M = pd.DataFrame(rows)
    M.to_csv(f"{a.out}/metrics.csv", index=False)
    C = pd.concat(curves); C.to_csv(f"{a.out}/pr_curves.csv", index=False)

    # ---- printed summary
    md = []
    for level in ("service", "visit"):
        for key in ("corrected", "original"):
            d = M[(M.level == level) & (M["labels"] == key)]
            o = d[d.point == "overall"].set_index("model")[["n", "PA", "AUC_PR", "AUC_ROC"]]
            parts = [o]
            for pt in ("recall 75%", "recall 80%", "cut-off 0.8"):
                x = d[d.point == pt].set_index("model")
                cols = ["cutoff", "TP", "FP", "FN", "TN", "precision"] + (["recall"] if pt == "cut-off 0.8" else [])
                parts.append(x[cols].add_prefix(f"{pt} | "))
            T = pd.concat(parts, axis=1).T
            frac = T.index.str.split("|").str[-1].str.contains("precision|recall|AUC|cutoff")
            T = T.astype(float).apply(lambda r: r.map(("{:.4f}" if frac[T.index.get_loc(r.name)] else "{:,.0f}").format), axis=1)
            title = f"{level.upper()} level, {key} labels"
            print(f"\n== {title}\n" + T.to_string())
            md.append(f"## {title}\n\n" + T.to_string() + "\n")
    open(f"{a.out}/summary.md", "w").write("\n".join(md))

    # ---- data funnel + split counts of the final run
    final = runs.get("pax_s42") or list(runs.values())[-1]
    if os.path.exists(f"{final}/removal_funnel.csv"):
        print(f"\n== data funnel ({final})\n" + pd.read_csv(f"{final}/removal_funnel.csv").to_string(index=False))
    if os.path.exists(f"{final}/split_counts.json"):
        print(f"\n== train / validation / test counts ({final})\n" + json.dumps(json.load(open(f"{final}/split_counts.json")), indent=1))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        for n, g in C.groupby("model", sort=False):
            if n.endswith("s42"):
                ax.plot(g.recall, g.precision, label=n)
        ax.axvline(0.8, ls=":", c="grey"); ax.axvline(0.75, ls=":", c="grey")
        ax.set_xlabel("recall (share of PA services caught)"); ax.set_ylabel("precision (share of flags that are PA)")
        ax.set_title("Service level, corrected labels, seed 42"); ax.legend(); ax.grid(alpha=0.3)
        fig.savefig(f"{a.out}/pr_service_corrected.png", dpi=150, bbox_inches="tight")
    except Exception as e:                                   # plot is optional
        print("plot skipped:", e)
    print(f"\noutputs -> {a.out}/ (metrics.csv, summary.md, pr_curves.csv, pr_service_corrected.png)")


if __name__ == "__main__":
    main()
