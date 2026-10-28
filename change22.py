import pandas as pd, pyarrow.fs as pafs
p = "cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v4/ai_prior_auth_09282026_v3_2026_std.parquet"
d = pd.read_parquet(p, columns=["PAYOR","PLAN","PRODUCT_ID","PRODUCT_NAME"], filesystem=pafs.S3FileSystem())
print(d.PRODUCT_NAME.nunique(), d.PRODUCT_NAME.isna().mean())
print(d.PRODUCT_NAME.value_counts().head(25).to_string())
print(d.groupby(["PAYOR","PLAN"], observed=True).PRODUCT_NAME.nunique().describe())
