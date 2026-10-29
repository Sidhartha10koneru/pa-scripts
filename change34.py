import pandas as pd, numpy as np, pyarrow.fs as pafs
p = "cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v4/ai_prior_auth_09282026_v3_2025_std.parquet"
cols = ["CSN","FINAL_CPT","GROUP_LABEL","LINE_AUTHORIZATION_NUMBER","CARC","FILING_ORDER_C"]
d = pd.read_parquet(p, columns=cols, filesystem=pafs.S3FileSystem())
d = d[pd.to_numeric(d.FILING_ORDER_C, errors="coerce") == 1].copy()
d["lab"] = d.GROUP_LABEL.astype(str).str.upper()
d = d[d.lab.isin(["TRUE","FALSE"])]
d["pa"] = d.lab.eq("TRUE")
d["FINAL_CPT"] = d.FINAL_CPT.astype(str)

# 1) 99214 alone vs with other services in the visit
n_other = d.groupby("CSN").FINAL_CPT.transform(lambda s: (s != "99214").sum())
e = d[d.FINAL_CPT == "99214"].assign(alone=lambda x: n_other[x.index] == 0)
print("1) 99214 PA rate | alone:", f"{e[e.alone].pa.mean():.2%} ({e.alone.sum():,} lines)",
      "| with other services:", f"{e[~e.alone].pa.mean():.2%} ({(~e.alone).sum():,} lines)")

# 2) what else is on visits where 99214 is TRUE
true_csn = e.loc[e.pa, "CSN"].unique()
other = d[d.CSN.isin(true_csn) & (d.FINAL_CPT != "99214")]
print(f"\n2) {len(true_csn):,} visits with 99214 TRUE; share that have another service: "
      f"{other.CSN.nunique() / len(true_csn):.1%}\n   most common other CPTs:\n",
      other.FINAL_CPT.value_counts().head(15).to_string())

# 3) is the auth number on 99214 the same as on other lines of the same visit?
a = d[d.LINE_AUTHORIZATION_NUMBER.notna() & (d.LINE_AUTHORIZATION_NUMBER.astype(str).str.strip() != "")]
auth_99214 = a[a.FINAL_CPT == "99214"][["CSN","LINE_AUTHORIZATION_NUMBER"]].drop_duplicates()
auth_other = a[a.FINAL_CPT != "99214"][["CSN","LINE_AUTHORIZATION_NUMBER"]].drop_duplicates()
m = auth_99214.merge(auth_other, on=["CSN","LINE_AUTHORIZATION_NUMBER"], how="left", indicator=True)
print(f"\n3) 99214 lines with an auth number: {len(auth_99214):,}; "
      f"same auth number also on another line of the visit: {(m._merge == 'both').mean():.1%}")

# 4) missing-auth denials on 99214 lines themselves
codes = d.CARC.astype(str).str.extract(r"(\d+)")[0]
auth_denial = codes.isin(["15","197","198"])
print(f"\n4) missing-auth denial codes (CARC 15/197/198) on 99214 lines: {auth_denial[d.FINAL_CPT=='99214'].sum():,} "
      f"of {(d.FINAL_CPT=='99214').sum():,} | on all other lines: {auth_denial[d.FINAL_CPT!='99214'].sum():,}")
