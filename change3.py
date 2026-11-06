import pandas as pd, pyarrow.fs as pafs
fs = pafs.S3FileSystem()
ACUITY = ["15","16","39","59","61","62","96","109","147","165","197","198","210","243","A1","P17","W6"]
OURS   = ["15","39","197","198","210","284","296","302"]                     # current Level-2 CARC list
ALL = sorted(set(ACUITY + OURS), key=lambda c: (c[0].isalpha(), c.zfill(4)))
rows, anyhit = [], []
for y in [2025, 2026]:
    p = f"cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v4/ai_prior_auth_09282026_v3_{y}_std.parquet"
    d = pd.read_parquet(p, filesystem=fs, columns=["CARC", "GROUP_LABEL"])
    d["L"] = d.GROUP_LABEL.astype(str).str.strip().str.upper().map({"TRUE": "TRUE", "1": "TRUE", "FALSE": "FALSE", "0": "FALSE"}).fillna("AMBIGUOUS")
    c = d.CARC.astype(object).fillna("").astype(str).str.upper()
    d = d[c.ne("") & c.ne("NAN")]
    codes = d.CARC.astype(str).str.upper().str.findall(r"\b[A-Z]?\d+\b")   # exact tokens: 'MA15' does not match '15'
    e = d[["L"]].assign(code=codes).explode("code").reset_index().drop_duplicates(["index", "code"])
    e = e[e.code.isin(ALL)]
    t = pd.crosstab(e.code, e.L).reindex(ALL).fillna(0).astype(int)
    for k in ["TRUE", "FALSE", "AMBIGUOUS"]:
        t[k] = t.get(k, 0)
    t["lines"] = t[["TRUE", "FALSE", "AMBIGUOUS"]].sum(1)
    t["not_TRUE_today"] = t.FALSE + t.AMBIGUOUS                              # would become TRUE if the code counts as PA
    t["in_acuity"] = t.index.isin(ACUITY); t["in_ours"] = t.index.isin(OURS); t["year"] = y
    rows.append(t.reset_index())
    h = e[e.code.isin(ACUITY)].drop_duplicates("index")                      # lines with ANY Acuity code
    anyhit.append({"year": y, **h.L.value_counts().to_dict()})
R = pd.concat(rows)[["year", "code", "in_acuity", "in_ours", "lines", "TRUE", "FALSE", "AMBIGUOUS", "not_TRUE_today"]]
pd.set_option("display.width", 200)
print(R.to_string(index=False))
print("\nlines with any Acuity code, by current label:\n", pd.DataFrame(anyhit).fillna(0).to_string(index=False))
R.to_csv("acuity_code_counts.csv", index=False)
