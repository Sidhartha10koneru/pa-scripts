"""
compare_quality_modes.py - compare models trained with quality codes kept / labelled FALSE / excluded.

Fair comparison = the services every model is scored on, with the SAME labels:
  A. non-quality services (labels identical in all three runs)    <- main result
  B. all services in each run's own test set (what each run reports by itself)
  C. visit level (CSN): visit PA if any service PA, score = max service score; non-quality services only

Each run's test_preds.parquet (pa_experiments output) is read; services = CSN + CPT, label = max, score = max.
Metrics: AUC-PR, AUC-ROC, and TP/FP/FN/TN at 80% and 90% recall (cut-off chosen on that scope's own PA services).

Run : python compare_quality_modes.py exp_qc_keep=keep exp_qc_false=false exp_qc_exclude=exclude --out qc_compare
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

QUALITY = r"^\d{4}F$"


def services(path):
    t = pd.read_parquet(path)
    t = t.loc[:, ~t.columns.duplicated()]
    p = [c for c in t.columns if c.startswith("p_")]
    if len(p) != 1:
        raise SystemExit(f"{path}: expected one score column p_*, found {p}")
    t["CSN"] = t.CSN.astype(str)
    t["FINAL_CPT"] = t.FINAL_CPT.astype(str).str.strip().str.upper()
    s = t.groupby(["CSN", "FINAL_CPT"]).agg(y=("y", "max"), score=(p[0], "max")).reset_index()
    s["q"] = s.FINAL_CPT.str.fullmatch(QUALITY)
    return s


def metrics(y, s, scope, run):
    y, s = np.asarray(y), np.asarray(s)
    r = {"run": run, "scope": scope, "n": len(y), "PA": int(y.sum()), "AUC_PR": average_precision_score(y, s),
         "AUC_ROC": roc_auc_score(y, s)}
    for rec in (0.80, 0.90):
        cut = np.quantile(s[y == 1], 1 - rec)
        f = s >= cut
        k = int(rec * 100)
        r |= {f"cut@{k}": cut, f"TP@{k}": int((f & (y == 1)).sum()), f"FP@{k}": int((f & (y == 0)).sum()),
              f"FN@{k}": int((~f & (y == 1)).sum()), f"TN@{k}": int((~f & (y == 0)).sum())}
        r[f"precision@{k}"] = r[f"TP@{k}"] / max(r[f"TP@{k}"] + r[f"FP@{k}"], 1)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="folder=label, e.g. exp_qc_keep=keep")
    ap.add_argument("--out", default="qc_compare")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    rows, keys = [], {}
    for item in a.runs:
        folder, label = item.split("=") if "=" in item else (item, item)
        s = services(f"{folder}/test_preds.parquet")
        keys[label] = set(zip(s.CSN, s.FINAL_CPT))
        nq = s[~s.q]
        rows.append(metrics(nq.y, nq.score, "A non-quality services", label))
        rows.append(metrics(s.y, s.score, "B all services (own test set)", label))
        v = nq.groupby("CSN").agg(y=("y", "max"), score=("score", "max"))
        rows.append(metrics(v.y, v.score, "C visits (non-quality services)", label))
        print(f"{label}: {len(s):,} services ({int(s.q.sum()):,} quality) | PA {int(s.y.sum()):,} "
              f"(quality PA {int(s[s.q].y.sum()):,})")
    nq_sets = [{k for k in v if not pd.Series([k[1]]).str.fullmatch(QUALITY).iloc[0]} for v in keys.values()]
    if len(nq_sets) > 1 and any(x != nq_sets[0] for x in nq_sets[1:]):
        print("!! WARNING: the non-quality services differ between runs - check that all runs used the same test window/filters")
    M = pd.DataFrame(rows)
    M.to_csv(f"{a.out}/compare.csv", index=False)
    cols = ["run", "n", "PA", "AUC_PR", "AUC_ROC", "TP@80", "FP@80", "FN@80", "precision@80", "FP@90", "FN@90", "precision@90"]
    for sc, g in M.groupby("scope"):
        print(f"\n== {sc}\n", g[cols].round(4).to_string(index=False))
    print(f"\noutput: {a.out}/compare.csv")


if __name__ == "__main__":
    main()
