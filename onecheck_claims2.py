m = pd.read_parquet(path, columns=["PAYOR","FINAL_CPT","GROUP_LABEL"], filesystem=s3)
m = m[m.PAYOR.astype(str) == "MEDICARE"]
m["cpt"] = m.FINAL_CPT.astype(str).str.strip().str.upper()
t = m[m.GROUP_LABEL.astype(str) == "TRUE"].cpt.value_counts()
f = m[m.GROUP_LABEL.astype(str) == "FALSE"].cpt.value_counts()
s = pd.DataFrame({"true_cov":  r.cpt_code.map(t).fillna(0).groupby(r.PAYOR_CODE).sum() / t.sum(),
                  "false_cov": r.cpt_code.map(f).fillna(0).groupby(r.PAYOR_CODE).sum() / f.sum()})
s["lift"] = s.true_cov / s.false_cov
print(s.sort_values("lift", ascending=False).head(10).round(3))
print(s.loc["CMSHSP"].round(3))
