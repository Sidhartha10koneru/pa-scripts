import re
col = [c for c in rules_df.columns if "Auth Requirement" in c][0]
coded = rules_df[rules_df.CODING_STATUS == "CODED_LIST"].copy()
t = coded[col].fillna("").astype(str)

flags = {
  "dx_exclusion":   t.str.contains(r"not required for the following diagnosis", case=False),
  "dx_inclusion":   t.str.contains(r"(only|for these) diagnosis|oncology diagnosis|cancer diagnosis|billed with one of the following Dx", case=False),
  "has_icd_codes":  t.str.contains(r"\b[A-TV-Z]\d{2}(?:\.\w{1,4})?\b"),
  "place_of_svc":   t.str.contains(r"place of service", case=False),
  "inpt_outpt":     t.str.contains(r"inpatient only|outpatient only|no prior authorization required for outpatient|inpatient and outpatient", case=False),
  "modifier":       t.str.contains(r"modifier", case=False),
  "plan_exclusion": t.str.contains(r"plan exclusion|ISNP|D-SNP|special needs", case=False),
  "group_numbers":  t.str.contains(r"group|policies", case=False) & t.str.contains(r"\b\d{5}\b"),
  "state_specific": t.str.contains(r"applicable to|only in|Virgin Island", case=False),
  "cost_threshold": t.str.contains(r"\$\s?\d", case=False),
  "notification":   t.str.contains(r"notification", case=False),
}
f = pd.DataFrame(flags)
f["any_condition"] = f.any(axis=1)
print("CODED_LIST rows:", len(f))
print((f.mean() * 100).round(1).sort_values(ascending=False).astype(str) + "%")
print("\nBy payor (share of rows with any condition):")
print(f.assign(PAYOR_CODE=coded.PAYOR_CODE.values).groupby("PAYOR_CODE").any_condition.mean()
        .sort_values(ascending=False).head(20).round(2))
