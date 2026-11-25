"""
referral_check.py - how much of our error is tied to referrals? (sizing for the team, counts only)

Flags each test service (CSN + CPT) that has referral evidence on any of its lines:
  REF_TEXT  auth-number text looks like a referral (RFRL / REFERRAL / REFRL / REF ...), incl. "RFRL REQD"
  REF_CODE  referral denial code 165 / 243 (Acuity list) in ACUITY_AUTH_CODES
  VLA       VISIT_LEVEL_AUTH = 1 (TRUE via a shared auth number with no owner line - kept TRUE pending the referral decision)
Then, at the cut-off for --recall on ALL test services (model not retrained), reports:
  0  TP / FP / FN / TN for all services and E/M, split by referral evidence
  1  Scenario "exclude referral services" - drop every flagged service (TP, FP, FN, TN alike), same cut-off and re-tuned cut-off
  2  Scenario "referral counts as PA"     - flagged non-PA services become PA
These are WHAT-IFs on a fixed model: a sizing, not a retrained result.
Raw auth text is never printed. Output = counts only.

  python referral_check.py --run exp_feat_core_s42 --out referral_core.csv
"""
import argparse
import os

import numpy as np
import pandas as pd
import pyarrow.fs as pafs

from build_pa_history import family

PATH = "cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v8_draft/ai_prior_auth_09282026_v3_2026_std.parquet"
REF_RE = r"\bRFRL\b|REFERR|REFRL|\bREF\b|\bRFL\b|REFERAL"


def key(s):
    return s.astype(object).fillna("").astype(str).str.replace(r"\.0$", "", regex=True).str.strip().str.upper()


def cm_counts(y, s, cut):
    f = s >= cut
    tp, fp = int((y & f).sum()), int((~y & f).sum())
    fn, tn = int((y & ~f).sum()), int((~y & ~f).sum())
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, services=len(y),
                precision=round(tp / max(tp + fp, 1), 4), recall=round(tp / max(tp + fn, 1), 4))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=PATH)
    ap.add_argument("--recall", type=float, default=0.85)
    ap.add_argument("--auth-col", default="LINE_AUTHORIZATION_NUMBER")
    ap.add_argument("--code-col", default="ACUITY_AUTH_CODES")
    ap.add_argument("--ref-codes", nargs="+", default=["165", "243"])
    a = ap.parse_args()

    # our test services
    t = pd.read_parquet(f"{a.run}/test_preds.parquet")
    t = t.loc[:, ~t.columns.duplicated()]
    p = [c for c in t.columns if c.startswith("p_")][0]
    t["k_csn"], t["k_cpt"] = key(t.CSN), key(t.FINAL_CPT)
    S = t.groupby(["k_csn", "k_cpt"]).agg(y=("y", "max"), s=(p, "max")).reset_index()
    S["y"] = S.y.astype(bool)
    S["fam"] = S.k_cpt.map(family)

    # referral evidence from the data file
    fs = None if os.path.exists(a.data) else pafs.S3FileSystem()
    import pyarrow.parquet as pq
    cols = pq.ParquetFile(fs.open_input_file(a.data) if fs else a.data).schema_arrow.names
    has = {c: c in cols for c in [a.auth_col, a.code_col, "VISIT_LEVEL_AUTH"]}
    print("columns found: " + " | ".join(f"{c} {'yes' if v else 'NO'}" for c, v in has.items()))
    use = ["CSN", "FINAL_CPT"] + [c for c, v in has.items() if v]
    d = pd.read_parquet(a.data, columns=use, filesystem=fs)
    d["k_csn"], d["k_cpt"] = key(d.CSN), key(d.FINAL_CPT)
    d = d[d.k_csn.isin(set(S.k_csn))]
    d["REF_TEXT"] = key(d[a.auth_col]).str.contains(REF_RE, regex=True) if a.auth_col in d else False
    codes = set(a.ref_codes)
    d["REF_CODE"] = (key(d[a.code_col]).str.replace(" ", "").str.split(",").map(lambda l: bool(codes & set(l)))
                     if a.code_col in d else False)
    d["VLA"] = pd.to_numeric(d.VISIT_LEVEL_AUTH, errors="coerce").fillna(0).eq(1) if "VISIT_LEVEL_AUTH" in d else False
    F = ["REF_TEXT", "REF_CODE", "VLA"]
    ev = d.groupby(["k_csn", "k_cpt"])[F].max().reset_index()
    S = S.merge(ev, on=["k_csn", "k_cpt"], how="left")
    S[F] = S[F].fillna(False).astype(bool)
    S["REF"] = S[F].any(axis=1)

    cut = float(np.quantile(S.s[S.y], 1 - a.recall))
    S["cm"] = np.select([S.y & (S.s >= cut), ~S.y & (S.s >= cut), S.y], ["TP", "FP", "FN"], "TN")
    rows = []

    # 0 where the referral evidence sits
    for scope, m in [("all", S.index == S.index), ("E/M", S.fam == "EM")]:
        x = S[m]
        rows.append(dict(table="0_current", scope=scope, **cm_counts(x.y, x.s, cut)))
        for ev_col in ["REF"] + F:
            c = x[x[ev_col]].cm.value_counts()
            rows.append(dict(table="0_with_evidence", scope=f"{scope} | {ev_col}",
                             **{k: int(c.get(k, 0)) for k in ["TP", "FP", "FN", "TN"]}, services=int(x[ev_col].sum())))

    # 1 exclude referral services (every class)
    for scope, m in [("all", S.index == S.index), ("E/M", S.fam == "EM")]:
        x = S[m & ~S.REF]
        rows.append(dict(table="1_exclude_same_cut", scope=scope, **cm_counts(x.y, x.s, cut)))
    x = S[~S.REF]
    cut1 = float(np.quantile(x.s[x.y], 1 - a.recall))
    rows.append(dict(table=f"1_exclude_recut_{a.recall:.0%}", scope="all", **cm_counts(x.y, x.s, cut1)))

    # 2 referral counts as PA
    y2 = S.y | S.REF
    rows.append(dict(table="2_referral_is_PA_same_cut", scope="all", **cm_counts(y2, S.s, cut)))
    m = S.fam == "EM"
    rows.append(dict(table="2_referral_is_PA_same_cut", scope="E/M", **cm_counts(y2[m], S.s[m], cut)))

    R = pd.DataFrame(rows)
    R.insert(2, "cut", round(cut, 6))
    R.loc[R.table.str.startswith("1_exclude_recut"), "cut"] = round(cut1, 6)
    R.to_csv(a.out, index=False)
    pd.set_option("display.width", 200)
    print(R.to_string(index=False))
    print(f"-> {a.out}  (what-ifs on a fixed model; counts only)")
    os._exit(0)


if __name__ == "__main__":
    main()
