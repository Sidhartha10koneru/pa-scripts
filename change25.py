i = mid.index[mid.FINAL_CPT.eq("99214")][0]
w = Xm.loc[[i]].copy(); w["RULE_PLAN_SHARE"] = 0.0
print(round(m.predict(Xm.loc[[i]])[0], 3), "->", round(m.predict(w)[0], 3))
# and what does the Healthfirst rule say for 99214?
r = pd.read_parquet("rules_analysis/all_rules.parquet", columns=["XREF_ID","PLAN_NAME","MEDICAL_CODES","EXCEPTION_CODES"])
print(r[(r.MEDICAL_CODES.astype(str) == "99214") & r.XREF_ID.astype(str).str.contains("HFST|HEALTHFIRST", case=False)]
      .drop_duplicates().to_string())
