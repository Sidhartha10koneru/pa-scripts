import pyarrow.parquet as pq
raw = "cli-ml-data-prod/datalake/ai-prior-auth/09282026/v3/ai_prior_auth_09282026_v3_2026.parquet"
names = pq.read_schema(raw, filesystem=fs).names
cols = [c for c in names if any(k in c.upper() for k in ["CARC", "RARC", "REMARK", "ADJ", "REASON"])]
print(cols)
r = pq.ParquetFile(raw, filesystem=fs).read_row_group(0, columns=cols).to_pandas()
print(r.head(20).to_string())
