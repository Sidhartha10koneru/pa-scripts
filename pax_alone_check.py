import sys, numpy as np, pandas as pd
P = sys.argv[1] if len(sys.argv) > 1 else "s3://cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v8_draft/ai_prior_auth_09282026_v3_{y}_std.parquet"
fs = None
if P.startswith("s3://"):
    import pyarrow.fs as pafs
    fs, P = pafs.S3FileSystem(), P.replace("s3://", "")
T = {"TRUE", "1", "1.0"}
BLANK = {"", "NAN", "NONE", "NULL", "<NA>"}
NO_AUTH = r"(NO\s*(REF|REFERRAL|AUTH|PA|PRIOR)\b.*|NOT\s*REQ.*|NAR|NPN|NPAR|N/?A|NONE|0+)"   # same as build_v7_1.py
COLS = ["CSN", "FILING_ORDER_C", "FINAL_CPT", "PAYOR", "LINE_AUTHORIZATION_NUMBER", "LABEL_V6", "LABEL_V6_PAX", "IS_PASSENGER"]

z = pd.read_csv("pax_zero_pa_cpt_payer.csv").assign(rate=lambda x: x.pa_before / x.lines)
z = z[(z.rate >= 0.20) & (z.lines >= 20)]
combos = pd.MultiIndex.from_arrays([z.cpt.astype(str), z.payor.astype(str)])
checks, rows = [], []
for y in [2022, 2023, 2024, 2025, 2026]:
    d = pd.read_parquet(P.format(y=y), columns=COLS, filesystem=fs)
    d["cpt"] = d.FINAL_CPT.astype(str).str.strip().str.upper()
    d["payor"] = d.PAYOR.astype(str).str.upper().str.replace(r"\bAS OF\b.*$", "", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
    d["claim"] = d.CSN.astype(str).str.replace(r"\.0$", "", regex=True) + "|" + \
        pd.to_numeric(d.FILING_ORDER_C, errors="coerce").fillna(1).astype(int).astype(str)
    a = d.LINE_AUTHORIZATION_NUMBER.astype(object).fillna("").astype(str).str.strip().str.upper().str.rstrip(",. ")
    d["auth"] = a.where(~a.isin(BLANK) & ~a.str.fullmatch(NO_AUTH, na=False), "")
    d["n_cpt_claim"] = d.groupby("claim").cpt.transform("nunique")              # 1 = billed alone on its claim
    au = d[d.auth != ""][["claim", "auth", "cpt"]].drop_duplicates().groupby(["claim", "auth"]).cpt.nunique().rename("n_cpt_auth")
    d = d.merge(au, left_on=["claim", "auth"], right_index=True, how="left")
    d["n_cpt_auth"] = d.n_cpt_auth.fillna(0)                                    # >=2 = same auth on another CPT of the claim
    d["pa_before"] = d.LABEL_V6.astype(str).str.strip().str.upper().isin(T)
    d["pa_after"] = d.LABEL_V6_PAX.astype(str).str.strip().str.upper().isin(T)
    flip = pd.to_numeric(d.IS_PASSENGER, errors="coerce").fillna(0).eq(1)
    p = d[flip]
    checks.append({"year": y, "flipped": len(p), "flipped_alone_on_claim": int((p.n_cpt_claim == 1).sum()),
                   "flipped_auth_not_shared": int((p.n_cpt_auth < 2).sum()), "flipped_without_auth": int((p.auth == "").sum()),
                   "flipped_not_PA_before": int((~p.pa_before).sum())})
    k = d[pd.MultiIndex.from_arrays([d.cpt, d.payor]).isin(combos)].copy()
    k["where"] = np.where(k.n_cpt_claim == 1, "alone", "with other CPTs")
    rows.append(k.groupby(["cpt", "payor", "where"]).agg(lines=("pa_before", "size"), pa_before=("pa_before", "sum"),
                                                          pa_after=("pa_after", "sum")))
    print(y, "done")

C = pd.DataFrame(checks).astype({"year": str})
C = pd.concat([C, C.drop(columns="year").sum().to_frame().T.assign(year="total")], ignore_index=True)
print("\n== 1. every flipped line: alone / unshared auth / no auth / not PA before  (all should be 0)")
print(C.to_string(index=False))

R = pd.concat(rows).groupby(level=[0, 1, 2]).sum().unstack("where", fill_value=0)
R.columns = [f"{w} {m}" for m, w in R.columns]
R = R.reindex(columns=[c for c in ["alone lines", "alone pa_before", "with other CPTs lines", "with other CPTs pa_before",
                                   "with other CPTs pa_after"] if c in R.columns], fill_value=0)
R = R.sort_values(R.columns[-2], ascending=False)
print("\n== 2. the high-PA-rate combinations now at zero PA: billed alone vs with other CPTs, at THAT payer")
print(R.to_string())
R.to_csv("pax_alone_check_combos.csv"); C.to_csv("pax_alone_check_flipped.csv", index=False)
print("\nsaved: pax_alone_check_combos.csv, pax_alone_check_flipped.csv")
