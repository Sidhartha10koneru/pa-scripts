import pandas as pd, pyarrow.dataset as ds, pyarrow.fs as pafs
V8 = "cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v8_draft/ai_prior_auth_09282026_v3_{year}_std.parquet"
fs = pafs.S3FileSystem(); parts = []
for y in [2022, 2023, 2024, 2025, 2026]:
    x = ds.dataset(V8.format(year=y), filesystem=fs, format="parquet").to_table(
        columns=["FINAL_CPT", "SERVICEDATE", "LABEL_V6_PAX"]).to_pandas()
    c = x.FINAL_CPT.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    n = pd.to_numeric(c.where(c.str.fullmatch(r"\d{5}")), errors="coerce")
    m = n.between(70010, 79999)
    parts.append(pd.DataFrame({"cpt": c[m], "dos": pd.to_datetime(x.SERVICEDATE[m], errors="coerce"),
                               "pa": x.LABEL_V6_PAX[m].astype(str).str.upper().eq("TRUE")}))
R = pd.concat(parts)
R["window"] = pd.cut(R.dos, pd.to_datetime(["1900-01-01", "2026-02-01", "2026-05-01", "2100-01-01"]), right=False,
                     labels=["train", "update", "test"])
print("distinct radiology CPTs, all years:", R.cpt.nunique())
print(R.groupby("window", observed=True).cpt.nunique().rename("distinct CPTs"))
g = R.groupby("cpt").agg(lines=("pa", "size"), pa_lines=("pa", "sum"))
print("CPTs with at least one PA line:", int((g.pa_lines > 0).sum()))
print("CPTs with 20+ lines:", int((g.lines >= 20).sum()), "| covering", round(g.lines[g.lines >= 20].sum() / g.lines.sum(), 4), "of lines")
print("test CPTs never seen before test:", len(set(R.cpt[R.window == "test"]) - set(R.cpt[R.window != "test"])))
