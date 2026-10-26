import pandas as pd, pyarrow.fs as pafs
raw = "cli-ml-data-prod/datalake/ai-prior-auth/09282026/v3/ai_prior_auth_09282026_v3_2026.parquet"
r = pd.read_parquet(raw, columns=["CSN","LINE","SERVICEDATE","CARC","RARC","CLAIMSTATUS",
                                  "FINAL_ADJUDICATION_DATE","PAID_AMT"], filesystem=pafs.S3FileSystem())
r["month"] = pd.to_datetime(r.SERVICEDATE, errors="coerce").dt.to_period("M").astype(str)
blank = lambda s: s.isna() | (s.astype(str).str.strip().isin(["", "nan", "None"]))
r["no_codes"] = blank(r.CARC) & blank(r.RARC)
r["not_adjudicated"] = blank(r.FINAL_ADJUDICATION_DATE)

# 1) rows with no codes: how many were never adjudicated, by month?  (pending → labelled FALSE)
print(r[r.no_codes].groupby("month").not_adjudicated.mean().round(3).to_string())
print(pd.crosstab(r.month, r.CLAIMSTATUS.astype(str), normalize="index").round(3).to_string())

# 2) rows per claim line (remit rows) by month: do older lines have more rows?
print(r.groupby(["month","CSN","LINE"]).size().groupby("month").mean().round(2).to_string())

# 3) top CARC codes over ALL rows (not just the first per line)
print(r.CARC.astype(str).value_counts().head(25).to_string())
