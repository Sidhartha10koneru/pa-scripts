a = pd.read_parquet(path, columns=["PAYOR","PLAN","FINAL_CPT","GROUP_LABEL"], filesystem=s3)
a = a[a.PAYOR.astype(str).str.contains("AARP")]
a["cpt"] = a.FINAL_CPT.astype(str).str.strip().str.upper()
t = a[a.GROUP_LABEL.astype(str) == "TRUE"].cpt.value_counts()
f = a[a.GROUP_LABEL.astype(str) == "FALSE"].cpt.value_counts()
r = exploded_rules_v2[["PAYOR_CODE","cpt_code"]].drop_duplicates()
score = pd.DataFrame({
    "true_cov":  r.cpt_code.map(t).fillna(0).groupby(r.PAYOR_CODE).sum() / t.sum(),
    "false_cov": r.cpt_code.map(f).fillna(0).groupby(r.PAYOR_CODE).sum() / f.sum()})
score["lift"] = score.true_cov / score.false_cov
print(score.sort_values("true_cov", ascending=False).head(10).round(3))

both = set(r.loc[r.PAYOR_CODE.isin(["AARP","UHC"]), "cpt_code"])
print(a[(a.GROUP_LABEL.astype(str) == "TRUE") & ~a.cpt.isin(both)].cpt.value_counts().head(15))
print(a[a.GROUP_LABEL.astype(str) == "TRUE"].PLAN.value_counts().head(10))
