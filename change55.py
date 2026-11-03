import re, pandas as pd

pat = re.compile(r"^([A-Z][A-Z0-9]{2})(?:#(\d+))?-(\d+)(?:\.(\d+))?\s*/\s*(.*)$")

def hl7_long(path):
    w = pd.read_csv(path, dtype=str, keep_default_na=False)          # all text: no 2.03E+11
    w["msg"] = w.get("MSH-10 / Message Control ID", pd.Series(range(len(w)))).astype(str)
    long = w.melt(id_vars="msg", var_name="column", value_name="value")
    long = long[long.value.str.strip() != ""]
    parts = long.column.str.extract(pat)
    parts.columns = ["segment", "repeat", "field", "component", "name"]
    long = pd.concat([long, parts], axis=1)
    long["file"] = path
    return long

L = pd.concat([hl7_long(f) for f in ["1_flat 1.csv", "2_flat 1.csv", "3_flat 1.csv"]], ignore_index=True)
pd.set_option("display.width", 200); pd.set_option("display.max_rows", 500)

# 1) which segments each file has, and how many filled fields
print(L.groupby(["file", "segment"]).size().unstack(0, fill_value=0))
# 2) readable dump: one line per filled field
print(L.sort_values(["file", "msg", "segment", "repeat", "field"])[["file", "segment", "repeat", "column", "value"]].to_string(index=False))
L.to_csv("accureg_samples_long.csv", index=False)
