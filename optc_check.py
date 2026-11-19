"""
optc_check.py - what each model does with the lines option C changes, on the test set.

Groups of test services (CSN + CPT, quality codes left out), from the v9 files:
  protected   was a passenger in v8, kept PA by option C            (PAX_PROTECTED)
  passenger   still a passenger under option C                      (IS_PASSENGER_C)
  visit auth  kept PA as visit-level auth                           (VISIT_LEVEL_AUTH)
  other PA    PA in the original labels and in none of the above
For each model: the cut-off that catches 80% of PA services (original labels), and the share of each group above it.
Option C wants 'protected' high (they are PA again), 'passenger' low, 'other PA' not lower than before.

  python optc_check.py base=exp_v8d_grp_s42 l6pax=exp_v8d_l6pax_s42 l6paxc=exp_v9d_l6paxc_s42 --base base \
      --v9-path "s3://<agreed>/std_data_v9_draft/ai_prior_auth_09282026_v3_{year}_std.parquet" --out v9d_optc_check_s42.csv
"""
import argparse
import os

import numpy as np
import pandas as pd

QUALITY = r"^\d{4}F$"


def services(path):
    t = pd.read_parquet(path)
    t = t.loc[:, ~t.columns.duplicated()]
    p = [c for c in t.columns if c.startswith("p_")][0]
    t["CSN"] = t.CSN.astype(str).str.replace(r"\.0$", "", regex=True)
    t["FINAL_CPT"] = t.FINAL_CPT.astype(str).str.strip().str.upper()
    t = t[~t.FINAL_CPT.str.fullmatch(QUALITY)]
    return t.groupby(["CSN", "FINAL_CPT"]).agg(y=("y", "max"), s=(p, "max"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="name=run_folder ...")
    ap.add_argument("--base", required=True, help="name of the GROUP_LABEL run: its test labels are the original labels")
    ap.add_argument("--v9-path", required=True)
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--recall", type=float, default=0.80)
    ap.add_argument("--out", default="v9d_optc_check.csv")
    a = ap.parse_args()
    runs = dict(r.split("=", 1) for r in a.runs)
    S = {k: services(f"{v}/test_preds.parquet") for k, v in runs.items()}
    common = set.intersection(*[set(s.index) for s in S.values()])
    y = S[a.base].y.loc[sorted(common)]
    idx = y.index

    import pyarrow.dataset as ds
    import pyarrow.fs as pafs
    p = a.v9_path.format(year=a.year)
    fs = pafs.S3FileSystem() if p.startswith("s3://") else None
    csn = set(idx.get_level_values(0))
    f = ds.dataset(p.replace("s3://", ""), filesystem=fs, format="parquet").to_table(
        columns=["CSN", "FINAL_CPT", "PAX_PROTECTED", "IS_PASSENGER_C", "VISIT_LEVEL_AUTH"]).to_pandas()
    f["CSN"] = f.CSN.astype(str).str.replace(r"\.0$", "", regex=True)
    f = f[f.CSN.isin(csn)]
    f["FINAL_CPT"] = f.FINAL_CPT.astype(str).str.strip().str.upper()
    for c in ["PAX_PROTECTED", "IS_PASSENGER_C", "VISIT_LEVEL_AUTH"]:
        f[c] = pd.to_numeric(f[c], errors="coerce").fillna(0)
    F = f.groupby(["CSN", "FINAL_CPT"])[["PAX_PROTECTED", "IS_PASSENGER_C", "VISIT_LEVEL_AUTH"]].max().reindex(idx).fillna(0)
    grp = np.select([F.PAX_PROTECTED.eq(1), F.IS_PASSENGER_C.eq(1), F.VISIT_LEVEL_AUTH.eq(1), y.eq(1)],
                    ["protected", "passenger", "visit auth", "other PA"], "not PA")
    G = pd.Series(grp, index=idx)
    print(f"test services: {len(idx):,} | PA (original labels) {int(y.sum()):,} | groups: "
          + ", ".join(f"{k} {v:,}" for k, v in G.value_counts().items() if k != "not PA"))
    rows = []
    for k in runs:
        s = S[k].s.loc[idx]
        cut = float(np.quantile(s[y.eq(1)], 1 - a.recall))
        r = {"model": k, f"cut-off @{a.recall:.0%} (original labels)": round(cut, 4)}
        for g in ["protected", "passenger", "visit auth", "other PA"]:
            m = G.eq(g)
            r[f"{g} above cut-off"] = f"{(s[m] >= cut).mean():.1%}" if m.any() else "n/a"
        rows.append(r)
    R = pd.DataFrame(rows)
    print(R.to_string(index=False))
    R.to_csv(a.out, index=False)
    print(f"-> {a.out}")
    os._exit(0)


if __name__ == "__main__":
    main()
