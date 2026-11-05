"""
bootstrap_compare.py - is model B really better than model A on the test set, or could it be luck of the test sample?

Paired visit bootstrap, no retraining:
  1. keep the services both runs scored, non-quality only (same labels in both)
  2. resample VISITS (CSN) with replacement (Poisson(1) weights per visit), N times
  3. each time compute, for both models on the same resample: AUC-PR, FP / precision at 80% and 90% recall
  4. report the difference B - A: mean, 95% interval, share of resamples where B is better

If the 95% interval of the difference does not cross 0, the gain is not explained by test-sample noise.
(It does not cover training randomness - that is the seed rerun.)

Run: python bootstrap_compare.py exp_nofilter_2022 exp_qc_exclude_rv --n 200 --out qc_bootstrap
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

QUALITY = r"^\d{4}F$"


def services(path):
    t = pd.read_parquet(path)
    t = t.loc[:, ~t.columns.duplicated()]
    p = [c for c in t.columns if c.startswith("p_")]
    if len(p) != 1:
        raise SystemExit(f"{path}: expected one p_* column, found {p}")
    t["CSN"] = t.CSN.astype(str)
    t["FINAL_CPT"] = t.FINAL_CPT.astype(str).str.strip().str.upper()
    s = t.groupby(["CSN", "FINAL_CPT"]).agg(y=("y", "max"), score=(p[0], "max")).reset_index()
    return s[~s.FINAL_CPT.str.fullmatch(QUALITY)]


def at_recall(y, s, w, rec):
    """weighted FP and precision at the cut-off that catches `rec` of the (weighted) PA"""
    o = np.argsort(-s, kind="stable")
    yw, nw = (y * w)[o], ((1 - y) * w)[o]
    tp, fp = np.cumsum(yw), np.cumsum(nw)
    k = np.searchsorted(tp, rec * tp[-1])
    return fp[k], tp[k] / (tp[k] + fp[k])


def stats(y, s, w):
    r = {"AUC_PR": average_precision_score(y, s, sample_weight=w)}
    for rec in (0.80, 0.90):
        fp, pr = at_recall(y, s, w, rec)
        r[f"FP@{int(rec * 100)}"], r[f"precision@{int(rec * 100)}"] = fp, pr
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a", help="baseline folder (e.g. exp_nofilter_2022)")
    ap.add_argument("run_b", help="candidate folder (e.g. exp_qc_exclude_rv)")
    ap.add_argument("--n", type=int, default=200, help="bootstrap resamples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="bootstrap")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    A = services(f"{a.run_a}/test_preds.parquet")
    B = services(f"{a.run_b}/test_preds.parquet")
    m = A.merge(B, on=["CSN", "FINAL_CPT"], suffixes=("_a", "_b"))
    bad = int((m.y_a != m.y_b).sum())
    print(f"A {len(A):,} | B {len(B):,} | common non-quality services {len(m):,} | PA {int(m.y_a.sum()):,} | "
          f"label mismatches {bad}")
    m = m[m.y_a == m.y_b]
    y = m.y_a.to_numpy().astype(float)
    sa, sb = m.score_a.to_numpy(), m.score_b.to_numpy()
    visit = pd.factorize(m.CSN)[0]
    nv = visit.max() + 1

    full = {k: (stats(y, sa, np.ones_like(y))[k], stats(y, sb, np.ones_like(y))[k]) for k in stats(y, sa, np.ones_like(y))}
    rng = np.random.default_rng(a.seed)
    rows = []
    for i in range(a.n):
        w = rng.poisson(1.0, nv)[visit].astype(float)             # same visit weights for both models (paired)
        ra, rb = stats(y, sa, w), stats(y, sb, w)
        rows.append({f"{k}_a": ra[k] for k in ra} | {f"{k}_b": rb[k] for k in rb})
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{a.n}")
    R = pd.DataFrame(rows)
    R.to_csv(f"{a.out}/bootstrap_samples.csv", index=False)

    out = []
    for k, (va, vb) in full.items():
        d = R[f"{k}_b"] - R[f"{k}_a"]
        better = (d > 0).mean() if not k.startswith("FP") else (d < 0).mean()
        out.append({"metric": k, "A": va, "B": vb, "B-A": vb - va, "rel_change": (vb - va) / va,
                    "CI95_low": d.quantile(0.025), "CI95_high": d.quantile(0.975), "share_B_better": better})
    O = pd.DataFrame(out)
    O.to_csv(f"{a.out}/bootstrap_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"\n== {a.run_b} minus {a.run_a} ({a.n} paired visit resamples)\n", O.round(4).to_string(index=False))
    print("\nB is reliably better on a metric if its 95% interval does not cross 0 (FP: interval below 0).")


if __name__ == "__main__":
    main()
