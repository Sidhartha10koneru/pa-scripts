import re, pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 1000); pd.set_option("display.max_colwidth", 60)

masked = lambda v: bool(re.fullmatch(r"[xX\-\s,.()]+", v))          # 'xx', 'xxx-xxx-xxxx', 'xx,xx' = de-identified
L["masked"] = L.value.map(masked)

# 1) one row per field: in which files, how many values, masked or real, example values
prof = (L.groupby(["segment", "column"])
          .agg(files=("file", lambda s: ",".join(sorted({f[0] for f in s}))),     # 1/2/3 = which sample file
               values=("value", "size"),
               masked_share=("masked", "mean"),
               examples=("value", lambda s: " | ".join(pd.unique(s[~s.map(masked)])[:4])))
          .reset_index())
prof["masked_share"] = prof.masked_share.round(2)
prof.to_csv("accureg_field_profile.csv", index=False)

# 2) overview per segment
print(prof.groupby("segment").agg(fields=("column", "size"),
                                  real_fields=("masked_share", lambda m: int((m < 1).sum()))).to_string())

# 3) the fields we care about first, with real examples
for s in ["MSH", "EVN", "PV1", "PV2", "IN1", "IN2", "ROL", "DG1"]:
    print(f"\n===== {s}")
    print(prof[(prof.segment == s) & (prof.masked_share < 1)][["column", "files", "values", "examples"]].to_string(index=False))

# 4) OBX: what is being observed (identifier + description), not every answer
o = L[L.segment == "OBX"]
ids = o[o.column.str.contains(r"-3\.(1|2) /", regex=True)]
print("\n===== OBX identifiers\n", ids.groupby(["file", "value"]).size().sort_values(ascending=False).head(50).to_string())
