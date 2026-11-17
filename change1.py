python - <<'EOF'
import pyarrow.dataset as ds, pyarrow.fs as pafs, re
fs = pafs.S3FileSystem()
p = "cli-ml-data-prod/ai-prior-auth/accureg_de_id_data/UABHS_April2026_scrubbed_YOBretained.parquet"
d = ds.dataset(p, filesystem=fs, format="parquet")
print("rows:", f"{d.count_rows():,}")
print("\nschema:"); print(d.schema)
t = d.head(5).to_pandas()
print("\nfirst 5 rows, value lengths per column:")
print(t.apply(lambda s: s.astype(str).str.len()).to_string())
for c in t.columns:
    v = t[c].astype(str)
    if v.str.startswith("MSH|").any():
        print(f"\nHL7 column: {c}\nsegment layout of the first 3 messages (segment name: number of fields):")
        for m in v.head(3):
            segs = [s for s in re.split(r"\r\n|\r|\n", m) if s.strip()]
            print("  ", "  ".join(f"{s.split('|')[0]}:{len(s.split('|'))}" for s in segs))
        msh = re.split(r"\r\n|\r|\n", v.iloc[0])[0].split("|")
        print("MSH-9 message type:", msh[8] if len(msh) > 8 else "?", "| MSH-12 version:", msh[11] if len(msh) > 11 else "?")
        in1 = [s for s in re.split(r"\r\n|\r|\n", v.iloc[0]) if s.startswith("IN1")]
        print("IN1 fields 2-4, 8-9, 15 of the first message:", [[s.split("|")[i] if len(s.split("|")) > i else "" for i in (2, 3, 4, 8, 9, 15)] for s in in1])
    else:
        print(f"\n{c}: e.g. {v.head(2).str.slice(0, 60).tolist()}")
EOF
