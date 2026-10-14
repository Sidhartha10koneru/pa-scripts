import re
col = [c for c in rules_df.columns if "Auth Requirement" in c][0]
txt = rules_df[col].fillna("").astype(str)

prod = r"(Dual Complete|Nursing Home|Assisted Living|Erickson Advantage|Preferred Care (?:Network|Partners)|Peoples Health|Rocky Mountain|UHC West|Virgin Island|[A-Z]?-?SNP|HMO-POS|ISNP|D-SNP|C-SNP|retiree)"
found = txt.str.findall(prod, flags=re.I).explode().str.upper().value_counts()
print("Plan/product terms in rules text:\n", found.head(20))

claims_plans = pairs["PLAN"].astype(str).str.upper()
for term in found.index[:20]:
    hits = claims_plans[claims_plans.str.contains(re.escape(term), regex=True)]
    print(f"{term:25s} -> {hits.nunique():4d} claims plans, e.g. {hits.unique()[:3].tolist()}")

rule_nums = set(re.findall(r"\b\d{5,6}\b", " ".join(txt[txt.str.contains("group|polic", case=False)])))
claim_nums = set(re.findall(r"\b\d{5,6}\b", " ".join(claims_plans.unique())))
print("\nGroup/policy numbers in rules:", len(rule_nums), "| in claims plan names:", len(claim_nums),
      "| overlap:", sorted(rule_nums & claim_nums)[:30])
