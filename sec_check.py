"""
sec_check.py - services billed on BOTH a primary and a secondary claim: where does the PA sit, and what changes if
the service is scored from its primary line?  Built to answer "show me the cases" (Raghu).

Unit: service = CSN + CPT, test months, lines the model scored (test_preds.parquet). Per line, from the data file by
SRC_ROW: filing order, payor | plan, label(s), and whether an auth number is present (never printed - only present /
absent and whether primary and secondary carry the SAME number).

  S1_overview         services with primary only / primary + secondary / secondary only, PA rate
  S2_label_pattern    (primary + secondary services) primary label x secondary label
  S3_auth_pattern     auth number on primary? on secondary? same number on both?  x label pattern
  S4_flips            outcome with highest-line scoring vs primary-line scoring (each at its own --recall cut-off),
                      by label pattern: which false alarms disappear, which PA (if any) are lost
  S5_payer_pairs      top primary payer -> secondary payer pairs, with their label pattern
  S6_examples         up to --n examples per pattern: CSN, CPT, both payers, both labels, auth present / same,
                      primary score vs secondary score, outcome under each rule (CSN only; no patient fields)

  python sec_check.py --run exp_feat_core_s42 --out sec_core                 # all families
  python sec_check.py --run exp_feat_core_s42 --out sec_core_imag --family IMAG
Outputs stay on EFS (S6 has CSNs).
"""
import argparse
import os

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.fs as pafs

from build_pa_history import family

DATA = "s3://cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v8_draft/ai_prior_auth_09282026_v3_2026_std.parquet"
TRUE_V = {"TRUE", "1", "1.0"}
NO_AUTH = r"^(|NONE|NA|N/A|NAR|NULL|NAN|0+|NO AUTH.*|NOT REQ.*|N\.?A\.?|AUTH NOT REQ.*|NO PA.*|RFRL REQD)$"


def key(s):
    return s.astype(object).where(s.notna(), None).map(lambda v: "" if v is None else str(v)) \
        .str.strip().str.replace(r"^(\d+)\.0$", r"\1", regex=True)


def clean_uniques(s):
    codes, uniq = pd.factorize(s.astype(object).fillna(""))
    u = pd.Series(uniq, dtype=object).astype(str).str.upper().str.replace(r"\bAS OF\b.*$", "", regex=True) \
        .str.replace(r"\s+", " ", regex=True).str.strip()
    return pd.Series(u.values[codes], index=s.index)


def lab(v):
    return np.where(v, "PA", "not PA")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--test-year", type=int, default=2026)
    ap.add_argument("--family", nargs="*", default=[])
    ap.add_argument("--label", default="LABEL_V6_PAX", help="label shown per line (the run's own label is test_preds.y)")
    ap.add_argument("--recall", type=float, default=0.85)
    ap.add_argument("--n", type=int, default=15)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 45)

    t = pd.read_parquet(f"{a.run}/test_preds.parquet")
    t = t.loc[:, ~t.columns.duplicated()]
    pc = [c for c in t.columns if c.startswith("p_")][0]
    t["k_csn"], t["k_cpt"] = key(t.CSN), key(t.FINAL_CPT).str.upper()
    t["fam"] = t.k_cpt.map(family)
    if a.family:
        t = t[t.fam.isin(a.family)].copy()
    t["fo"] = key(t.FILING_ORDER_C)
    t["prim"] = t.fo.eq("1")

    # line details from the data file
    fs, p = (pafs.S3FileSystem(), a.data.replace("s3://", "")) if a.data.startswith("s3://") else (pafs.LocalFileSystem(), os.path.abspath(a.data))
    dset = ds.dataset(p, filesystem=fs, format="parquet")
    cols = [c for c in ["CSN", "PAYOR", "PLAN", a.label, "LINE_AUTHORIZATION_NUMBER"] if c in dset.schema.names]
    rows = pd.to_numeric(t.SRC_ROW, errors="coerce")
    if "SRC_YEAR" in t:
        rows = rows.where(pd.to_numeric(t.SRC_YEAR, errors="coerce") == a.test_year)
    ok = rows.notna()
    d = dset.to_table(columns=cols).take(rows[ok].astype(int).values).to_pandas()
    d.index = t.index[ok]
    match = (key(d.CSN).values == t.k_csn[ok].values).mean()
    print(f"{a.run}: lines {len(t):,} | fetched by SRC_ROW {ok.mean():.1%} | CSN match {match:.1%}")
    if match < 0.99:
        raise SystemExit("SRC_ROW does not line up with --data")
    t["payer"] = (clean_uniques(d.PAYOR) + " | " + clean_uniques(d.PLAN)).reindex(t.index)
    t["lab"] = key(d[a.label]).str.upper().isin(TRUE_V).reindex(t.index).fillna(False) if a.label in d else t.y.astype(bool)
    if "LINE_AUTHORIZATION_NUMBER" in d:
        au = key(d.LINE_AUTHORIZATION_NUMBER).str.upper().str.replace(r"[\s,.]+$", "", regex=True)
        au = au.where(~au.str.fullmatch(NO_AUTH) & au.str.contains(r"\d"), "")
        t["auth"] = au.reindex(t.index).fillna("")
    else:
        t["auth"] = ""
    del d

    # service level: primary side and secondary side separately
    g = ["k_csn", "k_cpt"]
    P = t[t.prim].groupby(g).agg(p_lab=("lab", "max"), p_score=(pc, "max"), p_payer=("payer", "first"),
                                 p_auth=("auth", lambda s: "|".join(sorted(set(s) - {""}))))
    Q = t[~t.prim].groupby(g).agg(s_lab=("lab", "max"), s_score=(pc, "max"), s_payer=("payer", "first"),
                                  s_fo=("fo", "min"), s_auth=("auth", lambda s: "|".join(sorted(set(s) - {""}))))
    S = t.groupby(g).agg(y=("y", "max"), fam=("fam", "first")).join(P, how="left").join(Q, how="left")
    S["y"] = S.y.astype(bool)
    S["has_p"], S["has_s"] = S.p_score.notna(), S.s_score.notna()
    S["kind"] = np.select([S.has_p & S.has_s, S.has_p], ["primary + secondary", "primary only"], "secondary only")
    S["s_max"] = S[["p_score", "s_score"]].max(axis=1)
    S["s_prim"] = S.p_score.where(S.has_p, S.s_score)
    for rule in ["max", "prim"]:
        cut = float(np.quantile(S[f"s_{rule}"][S.y], 1 - a.recall))
        f = S[f"s_{rule}"] >= cut
        S[f"cm_{rule}"] = np.select([S.y & f, ~S.y & f, S.y], ["TP", "FP", "FN"], "TN")
        print(f"cut-off at {a.recall:.0%} recall, {'highest line' if rule == 'max' else 'primary line'}: {cut:.4f} | "
              f"TP {(S[f'cm_{rule}'] == 'TP').sum():,} | FP {(S[f'cm_{rule}'] == 'FP').sum():,}")

    out = {}
    o = S.groupby("kind").agg(services=("y", "size"), PA=("y", "sum"))
    o["PA_rate"] = (o.PA / o.services).round(4)
    o["share"] = (o.services / o.services.sum()).round(4)
    out["S1_overview"] = o

    B = S[S.kind == "primary + secondary"].copy()
    B["pattern"] = "primary " + pd.Series(lab(B.p_lab.astype(bool)), index=B.index) + " / secondary " + \
        pd.Series(lab(B.s_lab.astype(bool)), index=B.index)
    s2 = B.pattern.value_counts().rename("services").to_frame()
    s2["share"] = (s2.services / len(B)).round(4)
    s2["secondary_scores_higher"] = B.groupby("pattern").apply(lambda z: round((z.s_score > z.p_score).mean(), 4))
    s2["mean_primary_score"] = B.groupby("pattern").p_score.mean().round(3)
    s2["mean_secondary_score"] = B.groupby("pattern").s_score.mean().round(3)
    out["S2_label_pattern"] = s2

    B["auth_p"] = np.where(B.p_auth.ne(""), "auth on primary", "no auth on primary")
    B["auth_s"] = np.where(B.s_auth.eq(""), "no auth on secondary",
                           np.where(B.apply(lambda r: bool(set(r.p_auth.split("|")) & set(r.s_auth.split("|"))) and r.p_auth != "", axis=1),
                                    "SAME auth on secondary", "different auth on secondary"))
    out["S3_auth_pattern"] = pd.crosstab([B.auth_p, B.auth_s], B.pattern, margins=True)

    out["S4_flips"] = pd.crosstab([B.pattern, B.cm_max.rename("highest line")], B.cm_prim.rename("primary line"), margins=True)
    out["S4b_flips_all_services"] = pd.crosstab(S.cm_max.rename("highest line"), S.cm_prim.rename("primary line"), margins=True)

    pp = B.groupby(["p_payer", "s_payer"]).agg(services=("y", "size"), primary_PA=("p_lab", "mean"), secondary_PA=("s_lab", "mean"),
                                              FP_highest=("cm_max", lambda s: int((s == "FP").sum())),
                                              FP_primary=("cm_prim", lambda s: int((s == "FP").sum())))
    out["S5_payer_pairs"] = pp.sort_values("services", ascending=False).head(15).round(3)

    ex = []
    picks = [("primary PA / secondary not PA", None), ("primary not PA / secondary not PA", ("FP", "TN")),
             ("primary not PA / secondary PA", None), ("primary PA / secondary PA", None)]
    for pat, flip in picks:
        z = B[B.pattern == pat]
        if flip:
            z = z[(z.cm_max == flip[0]) & (z.cm_prim == flip[1])]
        z = z.sort_values("s_score", ascending=False).head(a.n)
        ex.append(z.assign(example_of=pat + (" (false alarm removed)" if flip else "")))
    E = pd.concat(ex).reset_index()
    E["auth"] = E.auth_p + " / " + E.auth_s
    E = E[["example_of", "k_csn", "k_cpt", "fam", "p_payer", "s_payer", "s_fo", "auth", "p_score", "s_score", "cm_max", "cm_prim"]]
    E[["p_score", "s_score"]] = E[["p_score", "s_score"]].round(3)
    E.to_csv(f"{a.out}/S6_examples.csv", index=False)

    for k, v in out.items():
        v.to_csv(f"{a.out}/{k}.csv")
        print(f"\n== {k}\n{v.to_string()}")
    print(f"\n== S6_examples (first 5 per pattern; all in {a.out}/S6_examples.csv)")
    print(E.groupby("example_of").head(5).to_string(index=False))
    print(f"\noutputs -> {a.out}/  (S6 has CSNs: keep on EFS)")
    os._exit(0)


if __name__ == "__main__":
    main()
