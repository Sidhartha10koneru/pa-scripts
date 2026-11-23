"""
fp_review.py - what are the false alarms of the best model? Label noise or real model errors?

Takes one run (default exp_feat_core_s42), finds the cut-off that catches --recall (85%) of PA services
(services = CSN + CPT, quality codes out, disputed v8 passengers left out = 'agreed' services), and compares the
FALSE ALARMS above the cut-off with the PA services caught (TP) on flags read from the v9 file:

  visit_has_pa        another service of the same visit is PA                (passenger-like)
  auth_text           the line has any text in LINE_AUTHORIZATION_NUMBER     (label says it is not a genuine auth)
  acuity_auth_code    an Acuity auth-denial CARC is on the line              (would be PA under V7)
  v7_strict_pa        LABEL_V7_STRICT is TRUE                                (PA under the auth-only CARC list)
  visit_level_auth    kept PA as visit-level auth in v8 (VISIT_LEVEL_AUTH)
High shares among false alarms (vs. low in the real negatives) point to label noise, not model error.

Also: the top CPTs / visit types / payers among false alarms, and a review sample (--sample rows) WITHOUT raw auth
values or patient fields, for an SME to judge 'really PA or not'. Writes to --out (stays on EFS; do not share outside).

  python fp_review.py --run exp_feat_core_s42 --out fp_review_core
"""
import argparse
import os

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.fs as pafs

V9D = ("cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v9_draft/"
       "ai_prior_auth_09282026_v3_{year}_std.parquet")
QUALITY = r"^\d{4}F$"
TRUE_SET = {"TRUE", "1", "1.0"}
EMPTY = {"", "NAN", "NONE", "NULL", "<NA>"}
COLS = ["CSN", "FINAL_CPT", "LINE_AUTHORIZATION_NUMBER", "ACUITY_AUTH_CODES", "LABEL_V7_STRICT", "IS_PASSENGER",
        "VISIT_LEVEL_AUTH", "VISITTYPE", "DEPARTMENT_ID", "SPECIALTY"]


def csn(s):
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()


def filled(s):
    return ~s.astype(str).str.strip().str.upper().isin(EMPTY)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="exp_feat_core_s42")
    ap.add_argument("--recall", type=float, default=0.85)
    ap.add_argument("--years", nargs="+", type=int, default=[2026])
    ap.add_argument("--data", default=V9D, help="v9 file pattern ({year}); local path or bucket/key without s3://")
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--out", default="fp_review_core")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # ---- test services and scores of the run
    t = pd.read_parquet(f"{a.run}/test_preds.parquet")
    t = t.loc[:, ~t.columns.duplicated()]
    p = [c for c in t.columns if c.startswith("p_")][0]
    t["CSN"], t["FINAL_CPT"] = csn(t.CSN), t.FINAL_CPT.astype(str).str.strip().str.upper()
    t = t[~t.FINAL_CPT.str.fullmatch(QUALITY)]
    keep = [c for c in ["PAYOR_CLEAN", "PAYOR_PLAN_CLEAN", "PLACE_OF_SERVICE_CODE", "FINAL_MODIFIERS", "Primary_DX"] if c in t]
    S = t.groupby(["CSN", "FINAL_CPT"]).agg(y=("y", "max"), s=(p, "max"), **{c: (c, "first") for c in keep})

    # ---- flags from the v9 file (test visits only)
    want = set(S.index.get_level_values(0))
    parts = []
    for y in a.years:
        path = a.data.format(year=y)
        fs = pafs.S3FileSystem() if not os.path.exists(path) else None
        d = ds.dataset(path, filesystem=fs, format="parquet")
        cols = [c for c in COLS if c in d.schema.names]
        for b in d.to_batches(columns=cols, batch_size=1_000_000):
            x = b.to_pandas()
            x["CSN"] = csn(x.CSN)
            x = x[x.CSN.isin(want)]
            if len(x):
                parts.append(x)
    f = pd.concat(parts, ignore_index=True)
    f["FINAL_CPT"] = f.FINAL_CPT.astype(str).str.strip().str.upper()
    f["auth_text"] = filled(f.LINE_AUTHORIZATION_NUMBER).astype(int) if "LINE_AUTHORIZATION_NUMBER" in f else 0
    f["acuity_auth_code"] = filled(f.ACUITY_AUTH_CODES).astype(int) if "ACUITY_AUTH_CODES" in f else 0
    f["v7_strict_pa"] = f.LABEL_V7_STRICT.astype(str).str.strip().str.upper().isin(TRUE_SET).astype(int) \
        if "LABEL_V7_STRICT" in f else 0
    for c in ["IS_PASSENGER", "VISIT_LEVEL_AUTH"]:
        f[c] = pd.to_numeric(f.get(c, 0), errors="coerce").fillna(0).astype(int)
    g = f.groupby(["CSN", "FINAL_CPT"]).agg(
        auth_text=("auth_text", "max"), acuity_auth_code=("acuity_auth_code", "max"), v7_strict_pa=("v7_strict_pa", "max"),
        passenger=("IS_PASSENGER", "max"), visit_level_auth=("VISIT_LEVEL_AUTH", "max"),
        **{c: (c, "first") for c in ["VISITTYPE", "DEPARTMENT_ID", "SPECIALTY"] if c in f})
    S = S.join(g, how="left")
    for c in ["auth_text", "acuity_auth_code", "v7_strict_pa", "passenger", "visit_level_auth"]:
        S[c] = S[c].fillna(0).astype(int)
    print(f"test services {len(S):,} | matched in data {S.VISITTYPE.notna().mean() if 'VISITTYPE' in S else float('nan'):.1%}")

    # ---- agreed services, cut-off, groups
    S = S[S.passenger.eq(0)]
    cut = float(np.quantile(S.s[S.y.eq(1)], 1 - a.recall))
    flag = S.s.ge(cut)
    pa_in_visit = S.groupby(level=0).y.transform("sum")
    S["visit_has_pa"] = (pa_in_visit - S.y).gt(0).astype(int)
    S["group"] = np.select([flag & S.y.eq(1), flag & S.y.eq(0), ~flag & S.y.eq(1)], ["TP", "FP", "FN"], "TN")
    n = S.group.value_counts()
    print(f"agreed services {len(S):,} | cut-off {cut:.4f} @ {a.recall:.0%} recall | "
          + ", ".join(f"{k} {n.get(k, 0):,}" for k in ["TP", "FP", "FN", "TN"])
          + f" | precision {n.get('TP', 0) / max(n.get('TP', 0) + n.get('FP', 0), 1):.1%}")

    flags = ["visit_has_pa", "auth_text", "acuity_auth_code", "v7_strict_pa", "visit_level_auth"]
    share = S.groupby("group")[flags].mean().reindex(["TP", "FP", "FN", "TN"])
    share.insert(0, "services", S.group.value_counts().reindex(share.index))
    print("\n== Share of services with each flag (FP high and TN low -> those FPs look like label noise)")
    print((share.assign(**{c: share[c].map("{:.1%}".format) for c in flags})).to_string())
    share.to_csv(f"{a.out}/flag_shares.csv")

    fp = S[S.group.eq("FP")]
    any_flag = fp[["auth_text", "acuity_auth_code", "v7_strict_pa"]].max(axis=1).eq(1)
    print(f"\nFalse alarms with an auth sign on the line itself (auth text, Acuity auth code or V7-strict PA): "
          f"{int(any_flag.sum()):,} of {len(fp):,} ({any_flag.mean():.1%})")
    print(f"False alarms in a visit that has another PA service: {int(fp.visit_has_pa.sum()):,} ({fp.visit_has_pa.mean():.1%})")
    print(f"False alarms with neither: {int((~any_flag & fp.visit_has_pa.eq(0)).sum()):,} "
          f"({(~any_flag & fp.visit_has_pa.eq(0)).mean():.1%})  <- most likely real model errors")

    # ---- where the false alarms are
    for c in ["FINAL_CPT", "VISITTYPE", "PAYOR_CLEAN", "SPECIALTY"]:
        if c not in S.columns and c != "FINAL_CPT":
            continue
        R = S.reset_index().assign(is_fp=lambda r: r.group.eq("FP").astype(int), is_tp=lambda r: r.group.eq("TP").astype(int))
        tab = R.groupby(c).agg(false_alarms=("is_fp", "sum"), pa_caught=("is_tp", "sum"), services=("y", "size"),
                               pa_rate=("y", "mean")).sort_values("false_alarms", ascending=False)
        tab["share_of_all_FP"] = tab.false_alarms / max(len(fp), 1)
        tab.head(50).to_csv(f"{a.out}/top_fp_by_{c}.csv")
        print(f"\n== Top 10 {c} by false alarms (share of all FP; cumulative top 10 = {tab.share_of_all_FP.head(10).sum():.1%})")
        print(tab.head(10).assign(share_of_all_FP=tab.share_of_all_FP.head(10).map("{:.1%}".format),
                                  pa_rate=tab.pa_rate.head(10).map("{:.1%}".format)).to_string())

    # ---- review sample: highest-scoring false alarms + a random part (no raw auth values, no patient fields)
    top = fp.sort_values("s", ascending=False)
    k = a.sample // 2
    smp = pd.concat([top.head(k), top.iloc[k:].sample(min(a.sample - k, max(len(top) - k, 0)), random_state=1)])
    smp = smp.reset_index().drop(columns=["y", "group", "passenger"]).rename(columns={"s": "score"})
    smp.insert(0, "why_in_sample", ["top score"] * min(k, len(smp)) + ["random"] * (len(smp) - min(k, len(smp))))
    smp["SME_really_PA (Y/N/unsure)"] = ""
    smp["SME_comment"] = ""
    smp.to_csv(f"{a.out}/fp_review_sample.csv", index=False)
    print(f"\nSaved to {a.out}/: flag_shares.csv, top_fp_by_*.csv, fp_review_sample.csv ({len(smp)} rows for SME review)")
    os._exit(0)


if __name__ == "__main__":
    main()
