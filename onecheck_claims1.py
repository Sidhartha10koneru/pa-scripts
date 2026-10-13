d = pd.read_parquet(path, columns=["CSN","FINAL_CPT","GROUP_LABEL"], filesystem=s3)
d["cpt"] = d.FINAL_CPT.astype(str).str.strip().str.upper()
d["on_any"] = d.cpt.isin(set(r.cpt_code))
tr = d[d.GROUP_LABEL.astype(str) == "TRUE"]
print("TRUE lines on any rules list:", round(tr.on_any.mean() * 100, 1), "%")
print("TRUE claims with at least one listed line:", round(d[d.CSN.isin(tr.CSN)].groupby("CSN").on_any.any().mean() * 100, 1), "%")
print("TRUE lines that are Category II (…F):", round(tr.cpt.str.endswith("F").mean() * 100, 1), "%")
print("TRUE lines that are office visits (99202–99499):", round(tr.cpt.str.match(r"^99[2-4]\d\d$").mean() * 100, 1), "%")
print(tr.cpt.value_counts().head(20))
