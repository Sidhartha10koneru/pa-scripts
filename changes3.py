a = rules_df.groupby("PAYOR_CODE").agg(aliases=("PAYOR_ALIAS","nunique"), plans=("PLAN_NAME","nunique"))
print(a.sort_values("aliases", ascending=False).head(15))
print(rules_df.loc[rules_df.PAYOR_CODE.isin(["UHC","AETNA","CIGNA"]), ["PAYOR_CODE","PAYOR_ALIAS","PLAN_NAME"]].drop_duplicates().head(30).to_string())

snp = pairs["PLAN"].astype(str).str.upper().str.contains(r"\bD-?SNP\b|\bSNP\b|DUAL COMPLETE", regex=True)
print("SNP / Dual claims:", int(pairs.loc[snp, "n"].sum()), "| by rules payor:")
print(pairs[snp].groupby("rules_payor").n.sum().sort_values(ascending=False))
