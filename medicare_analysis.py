cms = set(exploded_rules_v2.loc[exploded_rules_v2.PAYOR_CODE == "CMSHSP", "cpt_code"])
m = pd.read_parquet(path, columns=["PAYOR","PLAN","FINAL_CPT","GROUP_LABEL"], filesystem=s3)
m = m[(m.PAYOR.astype(str) == "MEDICARE") & m.GROUP_LABEL.astype(str).isin(["TRUE","FALSE"])]
m["on_cms"] = m.FINAL_CPT.astype(str).str.strip().str.upper().isin(cms)
print(m.groupby(["PLAN","GROUP_LABEL"], observed=True).on_cms.agg(["size","mean"]).round(4))
