x = pd.read_parquet(path, columns=["PAYOR","PLAN","LINE_OF_BUSINESS"], filesystem=s3)
print(x[x.PAYOR.astype(str) == "MEDICARE"].groupby(["PLAN","LINE_OF_BUSINESS"], observed=True, dropna=False).size())


print(rules_df[rules_df.PAYOR_CODE == "CMSHSP"][["PAYOR_NAME","PLAN_NAME","CODING_STATUS","SERVICE_GROUP"]].drop_duplicates().to_string())
cms = set(exploded_rules_v2.loc[exploded_rules_v2.PAYOR_CODE == "CMSHSP", "cpt_code"])
print("codes on CMSHSP list:", len(cms))
print({c: c in cms for c in ["15823","64615","30400","36473","15830","22551","63650","64490"]})
