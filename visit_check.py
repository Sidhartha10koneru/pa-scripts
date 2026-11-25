"""
visit_check.py - backup to patient history: does the REST OF THE VISIT (other services on the same CSN) separate PA from
false alarms? Uses no patient history and no labels of other services - only which other codes are on the same visit.
Nothing is written to S3; checked against the core model's test results.

Per service (CSN + CPT, quality codes out), from the other services on the same visit:
  n_other            number of other distinct CPTs on the visit
  other_<FAMILY>     1 if another service on the visit is in that family (THERAPY, INFUSION, HCPCS_J, IMAG, SURG, LAB, EM, ...)
Prints, at --recall (85%) with the core model's cut-off:
  1. PA rate with vs without each co-service, overall and for E/M, therapy, infusion, drugs, imaging
  2. share of TP / FP / FN / TN with each co-service      -> would it separate PA from false alarms?
  3. per family of the service: false alarms with no co-service of interest vs PA caught with one

  python visit_check.py --run exp_feat_core_s42
Needs build_pa_history.py (for family()) in the same folder.
"""
import argparse
import os

import numpy as np
import pandas as pd
import pyarrow.fs as pafs

from build_pa_history import family

V8D = "cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v8_draft/ai_prior_auth_09282026_v3_2026_std.parquet"
QUALITY = r"^\d{4}F$"
FAMS = ["HCPCS_J", "INFUSION", "THERAPY", "IMAG", "SURG", "LAB", "EM", "CARDIO", "MED"]


def csn(s):
    return s.astype(str).str.replace(r"\.0$", "", regex=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="exp_feat_core_s42")
    ap.add_argument("--data", default=V8D)
    ap.add_argument("--recall", type=float, default=0.85)
    a = ap.parse_args()
    pd.set_option("display.width", 220)

    t = pd.read_parquet(f"{a.run}/test_preds.parquet")
    t = t.loc[:, ~t.columns.duplicated()]
    p = [c for c in t.columns if c.startswith("p_")][0]
    t["CSN"], t["FINAL_CPT"] = csn(t.CSN), t.FINAL_CPT.astype(str).str.strip().str.upper()
    t = t[~t.FINAL_CPT.str.fullmatch(QUALITY)]
    S = t.groupby(["CSN", "FINAL_CPT"]).agg(y=("y", "max"), s=(p, "max")).reset_index()
    want = set(S.CSN)

    fs = None if os.path.exists(a.data) else pafs.S3FileSystem()
    v = pd.read_parquet(a.data, columns=["CSN", "FINAL_CPT"], filesystem=fs)
    v["CSN"], v["FINAL_CPT"] = csn(v.CSN), v.FINAL_CPT.astype(str).str.strip().str.upper()
    v = v[v.CSN.isin(want) & ~v.FINAL_CPT.str.fullmatch(QUALITY)].drop_duplicates()
    fam_of = {c: family(c) for c in v.FINAL_CPT.unique()}
    v["fam"] = v.FINAL_CPT.map(fam_of)
    cnt = pd.crosstab(v.CSN, v.fam)                                     # services per family on each visit
    n_cpt = v.groupby("CSN").FINAL_CPT.nunique()

    S["family"] = S.FINAL_CPT.map(lambda c: fam_of.get(c) or family(c))
    S["n_other"] = S.CSN.map(n_cpt).fillna(1).astype(int) - 1
    for f in FAMS:
        on_visit = S.CSN.map(cnt[f]) if f in cnt else pd.Series(0, index=S.index)
        S[f"other_{f}"] = (on_visit.fillna(0) - (S.family == f).astype(int)).gt(0).astype(int)   # another service, not itself
    cut = float(np.quantile(S.s[S.y.eq(1)], 1 - a.recall))
    f_ = S.s.ge(cut)
    S["grp"] = np.select([f_ & S.y.eq(1), f_ & S.y.eq(0), ~f_ & S.y.eq(1)], ["TP", "FP", "FN"], "TN")
    cols = ["n_other>0"] + [f"other_{f}" for f in FAMS]
    S["n_other>0"] = S.n_other.gt(0).astype(int)
    print(f"test services {len(S):,} | PA {int(S.y.sum()):,} | cut-off {cut:.4f} | FP {int((S.grp == 'FP').sum()):,}")

    print("\n== 1. PA rate of the service: with vs without each co-service on the same visit")
    rows = []
    for fam_name in ["ALL", "EM", "THERAPY", "INFUSION", "HCPCS_J", "IMAG"]:
        D = S if fam_name == "ALL" else S[S.family == fam_name]
        for c in cols:
            w = D[D[c] == 1]
            rows.append({"service family": fam_name, "co-service": c, "services_with": len(w), "pa_rate_with": w.y.mean() if len(w) else np.nan,
                         "pa_rate_without": D[D[c] == 0].y.mean()})
    R = pd.DataFrame(rows)
    R = R[R.services_with >= 200]
    print(R.assign(pa_rate_with=R.pa_rate_with.map("{:.1%}".format), pa_rate_without=R.pa_rate_without.map("{:.1%}".format)).to_string(index=False))

    print(f"\n== 2. Share of services with each co-service, by outcome at {a.recall:.0%} recall")
    sh = S.groupby("grp")[cols].mean().reindex(["TP", "FP", "FN", "TN"])
    sh.insert(0, "services", S.grp.value_counts().reindex(sh.index))
    print(sh.assign(**{c: sh[c].map("{:.1%}".format) for c in cols}).to_string())

    print("\n== 3. Per family: false alarms on visits with NO other services vs PA caught on visits WITH other services")
    G = S.assign(FP=S.grp.eq("FP"), TP=S.grp.eq("TP"), alone=S.n_other.eq(0))
    g = G.groupby("family").apply(lambda x: pd.Series({"FP": int(x.FP.sum()), "FP_alone": int((x.FP & x.alone).sum()),
                                                        "TP": int(x.TP.sum()), "TP_alone": int((x.TP & x.alone).sum())}),
                                   include_groups=False)
    print(g.sort_values("FP", ascending=False).head(12).to_string())
    os._exit(0)


if __name__ == "__main__":
    main()
