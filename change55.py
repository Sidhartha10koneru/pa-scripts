import pandas as pd
a = pd.read_parquet("test_set_rebuilt.parquet")           # the 2026-only run (last one saved)
# rerun with --years 2025 2026 --out test_set_2y.parquet first, then:
b = pd.read_parquet("test_set_2y.parquet")
extra = set(a.CSN.astype(str)) - set(b.CSN.astype(str))
print(len(extra))                                          # expect 153
import pyarrow.fs as pafs
p = "cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v4/ai_prior_auth_09282026_v3_2025_std.parquet"
d = pd.read_parquet(p, filesystem=pafs.S3FileSystem(), columns=["CSN", "SERVICEDATE", "FILING_ORDER_C", "GROUP_LABEL"])
d = d[d.CSN.astype(str).isin(extra)]
d["SVC"] = pd.to_datetime(d.SERVICEDATE, errors="coerce")
print(d.groupby("CSN").SVC.min().describe())               # their earliest dates in the 2025 file
print(d.SVC.dt.to_period("M").value_counts().sort_index())
