import pandas as pd, numpy as np, pyarrow.fs as pafs
p = "cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v4/ai_prior_auth_09282026_v3_2025_std.parquet"
cols = ["CSN","FINAL_CPT","GROUP_LABEL","LINE_AUTHORIZATION_NUMBER","CARC","RARC","FILING_ORDER_C","PAYOR"]
raw = pd.read_parquet(p, columns=cols, filesystem=pafs.S3FileSystem())
raw["FINAL_CPT"] = raw.FINAL_CPT.astype(str)
raw["lab"] = raw.GROUP_LABEL.astype(str).str.upper()

# truly alone: the WHOLE visit (all labels, primary + secondary) has no CPT other than 99214
other_any = raw.groupby("CSN").FINAL_CPT.transform(lambda s: (s != "99214").sum())
prim = raw[pd.to_numeric(raw.FILING_ORDER_C, errors="coerce") == 1]
x = prim[(prim.FINAL_CPT == "99214") & prim.lab.isin(["TRUE","FALSE"])].copy()
x["alone_strict"] = other_any.loc[x.index] == 0
print("99214 PA rate | alone (strict, all lines/claims):", f"{x[x.alone_strict].lab.eq('TRUE').mean():.2%}",
      f"({x.alone_strict.sum():,} lines) | not alone: {x[~x.alone_strict].lab.eq('TRUE').mean():.2%}")

# where does TRUE come from, for strictly-alone 99214?
t = x[x.alone_strict & x.lab.eq("TRUE")].copy()
auth = t.LINE_AUTHORIZATION_NUMBER.astype(str).str.strip()
t["has_auth"] = t.LINE_AUTHORIZATION_NUMBER.notna() & ~auth.str.upper().isin(["", "NAN", "NONE", "NULL"])
carc = t.CARC.astype(str).str.extract(r"(\d+)")[0]
t["pa_code"] = carc.isin(["15","39","197","198","210","284","296","302"]) | t.RARC.astype(str).str.contains("M62|N54", na=False)
t["source"] = np.select([t.has_auth & t.pa_code, t.has_auth, t.pa_code], ["auth number + denial code", "auth number only", "denial code only"], "neither (check!)")
print("\nSource of TRUE on strictly-alone 99214:\n", t.source.value_counts().to_string())
print("\nTop payors for 'auth number only':\n", t.loc[t.source == "auth number only", "PAYOR"].value_counts().head(10).to_string())

# is the same auth number reused across different visits? (referral / standing auth)
a = prim[prim.LINE_AUTHORIZATION_NUMBER.notna()].astype({"LINE_AUTHORIZATION_NUMBER": str})
visits_per_auth = a.groupby("LINE_AUTHORIZATION_NUMBER").CSN.nunique()
reuse = t.loc[t.has_auth, "LINE_AUTHORIZATION_NUMBER"].astype(str).map(visits_per_auth)
print("\nAlone-99214 auth numbers used on >1 visit:", f"{(reuse > 1).mean():.1%}", "| median visits per number:", reuse.median())
