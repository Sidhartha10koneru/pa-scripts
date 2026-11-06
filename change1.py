import pandas as pd, pyarrow.dataset as ds, pyarrow.fs as pafs, pyarrow.parquet as pq
fs = pafs.S3FileSystem()
MIXED = ["15", "16", "59", "96", "109", "147", "A1"]
y = 2026
raw = f"cli-ml-data-prod/datalake/ai-prior-auth/09282026/v3/ai_prior_auth_09282026_v3_{y}.parquet"
std = f"cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v4/ai_prior_auth_09282026_v3_{y}_std.parquet"
KEY = ["CSN", "FINAL_CPT", "SERVICEDATE"]
print([c for c in KEY if c not in pq.read_schema(raw, filesystem=fs).names], "<- missing key columns in raw (should be [])")

R = ds.dataset(raw, filesystem=fs, format="parquet")
code = lambda t: t.CARC.astype(str).str.strip().str.upper()
# pass 1: claim lines that have a mixed code on any remit row
hit = R.to_table(columns=KEY + ["CARC"], filter=ds.field("CARC").isin(MIXED)).to_pandas()
keys = hit[KEY].drop_duplicates()
# pass 2: all remit rows of those claim lines
allrows = R.to_table(columns=KEY + ["CARC"], filter=ds.field("CSN").isin(keys.CSN.unique().tolist())).to_pandas()
allrows = allrows.merge(keys, on=KEY)
allrows["code"] = code(allrows)
# current label of each claim line (std)
S = pd.read_parquet(std, filesystem=fs, columns=KEY + ["GROUP_LABEL"])
S["L"] = S.GROUP_LABEL.astype(str).str.upper().map({"TRUE": "TRUE", "1": "TRUE", "FALSE": "FALSE", "0": "FALSE"}).fillna("AMBIGUOUS")
S = S.drop_duplicates(KEY)
rows = []
for m in MIXED:
    lines = allrows[allrows.code == m][KEY].drop_duplicates()
    co = allrows.merge(lines, on=KEY)
    co = co[co.code != m].drop_duplicates(KEY + ["code"]).merge(S[KEY + ["L"]], on=KEY, how="left")
    t = co.groupby("code").L.value_counts().unstack(fill_value=0)
    t["lines"] = t.sum(1); t["TRUE_rate"] = (t.get("TRUE", 0) / t.lines).round(3)
    print(f"\n== {m}: {len(lines):,} claim lines | top codes on the same line")
    print(t.sort_values("lines", ascending=False).head(12).to_string())
    rows.append(t.reset_index().assign(mixed=m))
pd.concat(rows).to_csv(f"mixed_cooccur_{y}.csv", index=False)
