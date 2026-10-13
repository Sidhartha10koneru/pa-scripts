rules_df.loc[rules_df.PAYOR_NAME.str.contains("New York|Molina|HealthPlus|Oscar|Medicaid", case=False), ["PAYOR_CODE","PAYOR_NAME"]].drop_duplicates()

rules_df[rules_df.PAYOR_CODE.isin(["UHC","AARP"])].groupby("PAYOR_CODE").PLAN_NAME.unique()
