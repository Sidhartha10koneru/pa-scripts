x = pd.read_parquet(path, columns=["PAYOR","PLAN","FINAL_CPT","GROUP_LABEL"], filesystem=s3)
x = x[x.PAYOR.astype(str).str.contains("AARP") & (x.GROUP_LABEL.astype(str) == "TRUE")]
cpt = x.FINAL_CPT.astype(str).str.strip().str.upper()
for code in ["AARP", "UHC"]:
    lst = set(exploded_rules_v2.loc[exploded_rules_v2.PAYOR_CODE == code, "cpt_code"])
    print(code, "| codes on list:", len(lst), "| AARP PA claims covered:", round(cpt.isin(lst).mean() * 100, 1), "%")
