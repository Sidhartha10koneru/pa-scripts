b = pd.read_parquet(path, columns=["CSN","PAYOR","PLAN","FINAL_CPT","GROUP_LABEL"], filesystem=s3)
b = b[b.PAYOR.astype(str).str.contains("AARP")]
b["cpt"] = b.FINAL_CPT.astype(str).str.strip().str.upper()
uhc = set(r.loc[r.PAYOR_CODE == "UHC", "cpt_code"])
b["on_list"] = b.cpt.isin(uhc)
tr = b[b.GROUP_LABEL.astype(str) == "TRUE"]
print("TRUE lines on UHC list:", round(tr.on_list.mean() * 100, 1), "%")
print("TRUE claims with at least one listed line:", round(b[b.CSN.isin(tr.CSN)].groupby("CSN").on_list.any().mean() * 100, 1), "%")
print("TRUE lines that are Category II (…F) codes:", round(tr.cpt.str.endswith("F").mean() * 100, 1), "%")
print(tr.groupby(tr.PLAN.astype(str).str.contains("OPTUM CARE")).on_list.mean().round(3))
