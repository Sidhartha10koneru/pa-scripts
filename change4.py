import pandas as pd, pyarrow.fs as pafs
fs = pafs.S3FileSystem()
MIXED = ["15", "16", "59", "96", "109", "147", "A1"]
out = []
for y in [2025, 2026]:
    p = f"cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v4/ai_prior_auth_09282026_v3_{y}_std.parquet"
    d = pd.read_parquet(p, filesystem=fs, columns=["CARC", "RARC", "GROUP_LABEL"])
    d["L"] = d.GROUP_LABEL.astype(str).str.upper().map({"TRUE": "TRUE", "1": "TRUE", "FALSE": "FALSE", "0": "FALSE"}).fillna("AMBIGUOUS")
    d["carc"] = d.CARC.astype(str).str.upper().str.findall(r"\b[A-Z]?\d+\b")
    d["rarc"] = d.RARC.astype(object).fillna("").astype(str).str.upper().str.findall(r"\b[A-Z]{1,2}\d+\b")
    e = d.explode("carc"); e = e[e.carc.isin(MIXED)].explode("rarc")
    e["rarc"] = e.rarc.fillna("(none)")
    out.append(e.assign(year=y)[["year", "carc", "rarc", "L"]])
E = pd.concat(out)
t = E.groupby(["carc", "rarc"]).L.value_counts().unstack(fill_value=0)
t["lines"] = t.sum(1)
t["TRUE_rate"] = (t.get("TRUE", 0) / t.lines).round(3)
pd.set_option("display.width", 200)
for c in MIXED:
    if c in t.index.get_level_values(0):
        print(f"\n== CARC {c}\n", t.loc[c].sort_values("lines", ascending=False).head(12).to_string())
t.reset_index().to_csv("mixed_carc_by_rarc.csv", index=False)
