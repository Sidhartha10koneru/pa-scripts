v = d[d.FINAL_CPT == "99214"].copy()
pl = pd.read_parquet(p, columns=["CSN","FINAL_CPT","PAYOR","PLAN","FILING_ORDER_C"], filesystem=pafs.S3FileSystem())
pl = pl[pd.to_numeric(pl.FILING_ORDER_C, errors="coerce") == 1].drop_duplicates("CSN")[["CSN","PAYOR","PLAN"]]
v = v.merge(pl, on="CSN", how="left")
v["denial"] = v.CARC.astype(str).str.extract(r"(\d+)")[0].isin(["15","197","198"])
g = v.groupby(["PAYOR","PLAN"], observed=True).agg(lines=("pa","size"), pa_rate=("pa","mean"), denial_rate=("denial","mean"))
print(g[g.lines >= 300].sort_values("pa_rate", ascending=False).head(20).round(3).to_string())
print(v.loc[v.pa, "LINE_AUTHORIZATION_NUMBER"].astype(str).str[:12].value_counts().head(15).to_string())   # auth number formats
