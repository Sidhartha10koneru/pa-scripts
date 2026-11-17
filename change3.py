python - <<'EOF'
import pyarrow.dataset as ds, pyarrow.fs as pafs, pandas as pd, re
fs = pafs.S3FileSystem()
p = "cli-ml-data-prod/ai-prior-auth/accureg_de_id_data/UABHS_April2026_scrubbed_YOBretained.parquet"
d = ds.dataset(p, filesystem=fs, format="parquet")
names = d.schema.names
seg = pd.Series([re.split(r"_", n)[0] for n in names]).value_counts()
print("rows:", f"{d.count_rows():,}", "| columns:", len(names))
print("segments (columns per segment):", ", ".join(f"{k}:{v}" for k, v in seg.items()))
ins = [n for n in names if re.match(r"(IN1|IN2|IN3|MSH_9|PV1_2$|PV1_20$)", n)]
print("\ninsurance / type columns:", ins)
t = d.head(20000, columns=ins).to_pandas()
for c in ins:
    s = t[c].dropna().astype(str)
    s = s[~s.isin(["", "None", "nan", '""'])]
    print(f"\n{c}: filled {len(s)/len(t):.0%}, distinct {s.nunique():,}")
    print(s.str.slice(0, 80).value_counts().head(8).to_string())
EOF
