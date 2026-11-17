python - <<'EOF'
import pyarrow.dataset as ds, pyarrow.fs as pafs
fs = pafs.S3FileSystem()
p = "cli-ml-data-prod/ai-prior-auth/accureg_de_id_data/UABHS_April2026_scrubbed_YOBretained.parquet"
d = ds.dataset(p, filesystem=fs, format="parquet")
print("rows:", f"{d.count_rows():,}", "| columns:", len(d.schema.names))
t = d.head(1000).to_pandas()
for c in t.columns:
    s = t[c].astype(str)
    print(f"{c:40s} {str(t[c].dtype):10s} median len {int(s.str.len().median()):>7,}  starts with MSH|: {s.str.startswith('MSH|').mean():.0%}")
EOF
