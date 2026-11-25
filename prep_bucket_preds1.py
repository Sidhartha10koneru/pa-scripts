"""
prep_bucket_preds.py - turn one of OUR runs into a prediction file for the teammate's make_buckets.py, radiology only.

Writes <out>.parquet with the columns make_buckets.py expects (already in its renamed form):
  csn, cpt, mod, pos, payor, plan   raw values from the data file (payor / plan cleaned exactly like make_buckets.clean_names)
  pa_label                          True / False (our run's test label, LABEL_V6_PAX)
  pred_proba                        our service score (max over the service's lines)
and prints the cut-off that gives --recall (default 89.48%, the teammate's radiology recall) on these services.

  python prep_bucket_preds.py --run exp_feat_core_s42 --out rad_core.parquet
  python make_buckets.py --predict rad_core.parquet --summary-out rad_core_summary.csv --threshold <printed cut-off>
"""
import argparse
import os

import numpy as np
import pandas as pd
import pyarrow.fs as pafs

from make_buckets import clean_names                     # the teammate's own cleaning, so keys match his training keys

PATH = "cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v8_draft/ai_prior_auth_09282026_v3_2026_std.parquet"
RAW = {"CSN": "csn", "FINAL_CPT": "cpt", "FINAL_MODIFIERS": "mod", "SERVICEDATE": "dos", "PAYOR": "payor", "PLAN": "plan",
       "PLACE_OF_SERVICE_CODE": "pos"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=PATH)
    ap.add_argument("--recall", type=float, default=0.8948)
    ap.add_argument("--cpt-from", type=int, default=70010)
    ap.add_argument("--cpt-to", type=int, default=79999)
    a = ap.parse_args()

    t = pd.read_parquet(f"{a.run}/test_preds.parquet")
    t = t.loc[:, ~t.columns.duplicated()]
    p = [c for c in t.columns if c.startswith("p_")][0]
    t["k_csn"] = t.CSN.astype(str).str.replace(r"\.0$", "", regex=True)
    t["k_cpt"] = t.FINAL_CPT.astype(str).str.strip().str.upper()
    num = pd.to_numeric(t.k_cpt.where(t.k_cpt.str.fullmatch(r"\d{5}")), errors="coerce")
    t = t[num.between(a.cpt_from, a.cpt_to)]
    S = t.groupby(["k_csn", "k_cpt"]).agg(pa_label=("y", "max"), pred_proba=(p, "max")).reset_index()
    S["pa_label"] = S.pa_label.astype(bool)

    fs = None if os.path.exists(a.data) else pafs.S3FileSystem()
    d = pd.read_parquet(a.data, columns=list(RAW), filesystem=fs).rename(columns=RAW)
    for c in d.columns:                                   # file stores some columns as category -> plain values (clean_names can't fillna a category)
        if isinstance(d[c].dtype, pd.CategoricalDtype):
            d[c] = d[c].astype(object)
    d["k_csn"] = d.csn.astype(str).str.replace(r"\.0$", "", regex=True)
    d["k_cpt"] = d.cpt.astype(str).str.strip().str.upper()
    d = d.merge(S[["k_csn", "k_cpt", "pa_label"]], on=["k_csn", "k_cpt"])        # only our radiology test services
    d["dos"] = pd.to_datetime(d.dos, errors="coerce")
    d = d.sort_values(["pa_label", "dos"], ascending=[False, False], kind="stable").drop_duplicates(["k_csn", "k_cpt"])
    d[["payor", "plan"]] = d[["payor", "plan"]].apply(clean_names)
    out = S.merge(d[["k_csn", "k_cpt", "csn", "cpt", "mod", "pos", "payor", "plan"]], on=["k_csn", "k_cpt"], how="left")
    out = out.drop(columns=["k_csn", "k_cpt"])
    cut = float(np.quantile(out.pred_proba[out.pa_label], 1 - a.recall))
    out.to_parquet(a.out, index=False)
    print(f"{a.run}: radiology services {len(out):,} | PA {int(out.pa_label.sum()):,} | keys found {out.payor.notna().mean():.1%}")
    print(f"cut-off for {a.recall:.2%} recall: {cut:.6f}  -> use --threshold {cut:.6f}")
    print(f"-> {a.out}")
    os._exit(0)


if __name__ == "__main__":
    main()
