import pandas as pd, pyarrow.fs as pafs
s3 = pafs.S3FileSystem()
p = "cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v4/ai_prior_auth_09282026_v3_2026_std.parquet"
d = pd.read_parquet(p, columns=["CSN", "GROUP_LABEL", "FILING_ORDER_C"], filesystem=s3)
d["GROUP_LABEL"] = d["GROUP_LABEL"].astype(str)
prim = pd.to_numeric(d["FILING_ORDER_C"], errors="coerce").eq(1)
t = d["GROUP_LABEL"].eq("TRUE")
lab = d["GROUP_LABEL"].isin(["TRUE", "FALSE"])
print("distinct CSN, all:", d.CSN.nunique())
print("distinct CSN with a usable label:", d.loc[lab, "CSN"].nunique())
print("distinct CSN with TRUE:", d.loc[t, "CSN"].nunique(), "| primary payer only:", d.loc[t & prim, "CSN"].nunique())
print("TRUE lines per TRUE CSN:", round(t.sum() / d.loc[t, "CSN"].nunique(), 2))
