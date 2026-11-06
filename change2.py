p = "cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v4/ai_prior_auth_09282026_v3_2026_std.parquet"
import pyarrow.parquet as pq
print([c for c in pq.read_schema(p, filesystem=fs).names if "RARC" in c.upper() or "REMARK" in c.upper() or "CARC" in c.upper()])
d = pd.read_parquet(p, filesystem=fs, columns=["CARC", "RARC"])
m = d.CARC.astype(str).str.contains(r"\b(16|59|96|109|147|A1)\b")
print("RARC non-blank, all lines:", d.RARC.astype(str).str.strip().replace({"nan": "", "None": ""}).ne("").mean().round(3))
print("RARC non-blank, mixed-CARC lines:", d[m].RARC.astype(str).str.strip().replace({"nan": "", "None": ""}).ne("").mean().round(3))
print(d[m].RARC.astype(str).value_counts().head(15))
