print(rules_df.loc[rules_df.PAYOR_NAME.str.contains("New York|Molina|HealthPlus|Oscar|MetroPlus|Braven|Medicaid", case=False, na=False), ["PAYOR_CODE","PAYOR_NAME"]].drop_duplicates().to_string())
