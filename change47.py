"""
make_report.py - experiment sheet (CSV) in the same layout as experiment_summary_full.csv.

One row per run x model x level (service = CSN+CPT, visit = CSN), columns:
  experiment, level, sample, features, train_period, train_lines, train_visits, train_pa_rate,
  val_lines, val_visits, val_pa_rate, update_period, update_lines, update_visits, update_pa_rate,
  test_period, test_lines, test_visits, test_pa_rate, notes, AUC-ROC, AUC-PR, best_thr,
  Accuracy, Precision, Recall, F1, FP, FN, TP, TN, run
best_thr = the cut-off (0.05 .. 0.95 in steps of 0.05) with the highest F1; Accuracy..TN are at that cut-off.

Inputs per run folder: test_preds.parquet, split_info.json, split_counts.json
(split_counts.json is written by pa_experiments.py from now on; for older runs add --recount, which reloads
the data with the run's settings and recomputes train / validation sizes exactly as training split them).

Second file <out>_recall_targets.csv: the same models at 80 / 90 / 95% recall.

Run:
  python make_report.py exp_base_2025_2026=Base_2025_2026 exp_base_2022_2026=Base_2022_2026 \
      --out experiment_summary_base --recount
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import pa_base_adapter as pb

FEATURES = {"lookup": "rate per cell with fallback", "lgb": "CPT, mod, POS, payor, payor+plan",
            "lgb_rules": "CPT, mod, POS, payor, payor+plan, rules"}
COLS = ["experiment", "level", "sample", "features", "train_period", "train_lines", "train_visits", "train_pa_rate",
        "val_lines", "val_visits", "val_pa_rate", "update_period", "update_lines", "update_visits", "update_pa_rate",
        "test_period", "test_lines", "test_visits", "test_pa_rate", "notes", "AUC-ROC", "AUC-PR", "best_thr",
        "Accuracy", "Precision", "Recall", "F1", "FP", "FN", "TP", "TN", "run"]


def at_cut(y, p, cut):
    f = p >= cut
    tp, fp = int((f & (y == 1)).sum()), int((f & (y == 0)).sum())
    fn, tn = int((~f & (y == 1)).sum()), int((~f & (y == 0)).sum())
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"Accuracy": (tp + tn) / len(y), "Precision": prec, "Recall": rec,
            "F1": 2 * prec * rec / max(prec + rec, 1e-12), "FP": fp, "FN": fn, "TP": tp, "TN": tn}


def split_stats(df):
    """train / validation sizes exactly as train_lgb splits them (10% of visits, seed 42)"""
    from sklearn.model_selection import GroupShuffleSplit
    tr_i, va_i = next(GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42).split(df, groups=df[pb.CSN]))
    out = {}
    for n, x in (("train", df.iloc[tr_i]), ("val", df.iloc[va_i])):
        out |= {f"{n}_lines": len(x), f"{n}_visits": int(x[pb.CSN].nunique()), f"{n}_pa_rate": float(x.y.mean()),
                f"{n}_start": str(x.SVC_DATE.min().date()), f"{n}_end": str(x.SVC_DATE.max().date())}
    return out


def window_stats(x, n):
    return {f"{n}_lines": len(x), f"{n}_visits": int(x[pb.CSN].nunique()), f"{n}_pa_rate": float(x.y.mean()) if len(x) else None,
            f"{n}_period": f"{x.SVC_DATE.min():%Y-%m-%d} .. {x.SVC_DATE.max():%Y-%m-%d}" if len(x) else ""}


def recount(folder, info):
    """reload data with the run's settings and recompute train/val/test sizes (for runs without split_counts.json)"""
    import pyarrow.fs as pafs
    import pa_experiments as pe
    pb.DEDUP_KEEP_TRUE = info.get("dedup_keep_true", False)
    pb.INCLUDE_SECONDARY = info.get("include_secondary", False)
    raw = [c for c in pb.BASE_FEATS if c not in (pb.PAYOR_F, pb.PLAN_F)] + [pb.PAYOR_NAME, pb.PLAN_NAME]
    cols = list(dict.fromkeys([pb.CSN, pb.LABEL, pb.DATE, pb.FILING] + raw))
    d = pe.add_dates(pb.load(info["years"], cols, info.get("sample", 1.0), pafs.S3FileSystem()))
    train, update, test = pe.split(d, info["update_start"], info["test_start"], info.get("test_end"))
    if "IS_PRIMARY" in test:
        test = test[test.IS_PRIMARY == 1]
    full = pd.concat([train, update], ignore_index=True)
    st = {"frozen": split_stats(train), "full": split_stats(full)}
    t = window_stats(test, "test")
    json.dump({k: {**v, **t, "update_lines": 0, "update_visits": 0, "update_pa_rate": None, "update_period": ""}
               for k, v in st.items()}, open(f"{folder}/split_counts_recount.json", "w"), indent=1)
    return {k: {**v, **t, "update_lines": 0, "update_visits": 0, "update_pa_rate": None, "update_period": ""}
            for k, v in st.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run folders, optionally folder=label")
    ap.add_argument("--out", default="experiment_summary_new", help="output prefix")
    ap.add_argument("--recount", action="store_true", help="recompute train/val sizes for runs without split_counts.json")
    a = ap.parse_args()

    rows, rec_rows = [], []
    grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    for r in a.runs:
        folder, label = (r.split("=", 1) + [None])[:2]
        label = label or os.path.basename(folder.rstrip("/"))
        info = json.load(open(f"{folder}/split_info.json"))
        cpath = f"{folder}/split_counts.json"
        if os.path.exists(cpath):
            counts = json.load(open(cpath))
        elif os.path.exists(f"{folder}/split_counts_recount.json"):
            c2, counts = json.load(open(f"{folder}/split_counts_recount.json")), None
        elif a.recount:
            print(f"{folder}: no split_counts.json -> reloading data to recount (takes a few minutes)")
            c2 = recount(folder, info)
            counts = None
        else:
            print(f"{folder}: no split_counts.json (add --recount to fill train/val sizes)")
            counts, c2 = {}, {}
        p = pd.read_parquet(f"{folder}/test_preds.parquet")
        models = [c[2:] for c in p.columns if c.startswith("p_")]
        for m in models:
            cnt = counts.get(m, {}) if counts is not None else c2["full" if m.endswith("_full") else "frozen"]
            base = {"experiment": m, "sample": f"{info.get('sample', 1):.0%} of visits",
                    "features": FEATURES.get(m.rsplit("_", 1)[0], ""),
                    "train_period": f"{cnt.get('train_start', '')} .. {cnt.get('train_end', '')}" if cnt else "",
                    **{k: cnt.get(k) for k in ["train_lines", "train_visits", "train_pa_rate", "val_lines", "val_visits",
                                               "val_pa_rate", "update_period", "update_lines", "update_visits",
                                               "update_pa_rate", "test_period", "test_lines", "test_visits", "test_pa_rate"]},
                    "run": label}
            levels = {"service (CSN+CPT)": p.groupby([pb.CSN, pb.CPT], sort=False).agg(y=("y", "max"), s=(f"p_{m}", "max")),
                      "visit (CSN)": p.groupby(pb.CSN, sort=False).agg(y=("y", "max"), s=(f"p_{m}", "max"))}
            for lv, g in levels.items():
                y, s = g.y.values, g.s.values
                best = max(grid, key=lambda c: at_cut(y, s, c)["F1"])
                note = (f"{lv}: {len(g):,} units, {int(y.sum()):,} PA; secondary={info.get('include_secondary', False)}; "
                        f"repeat removal={'keep TRUE' if info.get('dedup_keep_true') else 'keep first'}; "
                        f"years={info.get('years')}")
                rows.append({**base, "level": lv, "notes": note, "AUC-ROC": roc_auc_score(y, s),
                             "AUC-PR": average_precision_score(y, s), "best_thr": best, **at_cut(y, s, best)})
                for t in (0.8, 0.9, 0.95):
                    cut = float(np.quantile(s[y == 1], 1 - t))
                    rec_rows.append({"run": label, "experiment": m, "level": lv, "recall_target": t, "cutoff": cut,
                                     "flagged_share": float((s >= cut).mean()), "AUC-PR": average_precision_score(y, s),
                                     **at_cut(y, s, cut)})
        print(f"read {folder} -> {label}: {len(models)} model(s)")

    out = pd.DataFrame(rows)[COLS]
    out.to_csv(f"{a.out}.csv", index=False)
    pd.DataFrame(rec_rows).to_csv(f"{a.out}_recall_targets.csv", index=False)
    pd.set_option("display.width", 250)
    print("\n", out[["run", "experiment", "level", "AUC-ROC", "AUC-PR", "best_thr", "Precision", "Recall", "F1",
                     "TP", "FP", "FN", "train_lines", "val_lines", "test_lines"]].round(4).to_string(index=False))
    print(f"\nWrote {a.out}.csv and {a.out}_recall_targets.csv")


if __name__ == "__main__":
    main()
