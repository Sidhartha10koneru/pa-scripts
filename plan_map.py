"""
plan_mapping.py - adds the rules PLAN to the payor master sheet.

Why: for UHC, Regence (ORRBCS), NM BCBS and Horizon (NJBCBS) the rules have
several coded plans with DIFFERENT CPT lists, so payor alone is not enough.
Every other payor keeps rules_plan = "ALL" (their plans share one list).

Needs in the notebook: rules_df and `pairs` (run run_mapping.py first).
Run:   %run -i plan_mapping.py
Adds column  rules_plan  to `pairs` and rewrites payor_master_review.xlsx.

rules_plan values:
  ALL            -> use every coded plan of the payor (plans share the same list)
  <plan name(s)> -> use only that rules plan's list ("||" between names if several)
  NO_CODED_PLAN  -> payor has coded plans, but none fits this claims plan (e.g. Horizon commercial)
  ""             -> payor itself is not a rules payor (NOT_IN_RULES, BLUECARD, FEP, ...)
"""
import re
import pandas as pd

if "pairs" not in globals():
    raise NameError("Run run_mapping.py first (it creates `pairs`).")

# coded plans per rules payor
cr = rules_df[rules_df["CODING_STATUS"].astype(str).str.strip() == "CODED_LIST"].copy()
cr["PAYOR_CODE"] = cr["PAYOR_CODE"].astype(str).str.strip()
cr["PLAN_NAME"] = cr["PLAN_NAME"].astype(str).str.strip()
coded_plans = cr.groupby("PAYOR_CODE")["PLAN_NAME"].agg(lambda s: sorted(set(s))).to_dict()

# ---------------------------------------------------------------------------
# (rules payor, pattern on claims "PAYOR | PLAN", which rules plan)
# Checked top to bottom within each payor; first hit wins.
# The last item is either a regex that finds the rules PLAN_NAME,
# or ALL / NO_CODED_PLAN.
# ---------------------------------------------------------------------------
PLAN_RULES = [
    # UnitedHealthcare: Commercial vs Medicare Advantage (only 29% of codes shared)
    ("UHC",    r"MEDICARE|ADVANTAGE|\bAARP\b|DUAL|\bSNP\b|\bMA\b", r"MEDICARE"),
    ("UHC",    r".",                                               r"COMMERCIAL"),
    # Regence Oregon: Commercial / Uniform / Medicare (FEP already tagged FEP)
    ("ORRBCS", r"MEDICARE|ADVANTAGE",                              r"MEDICARE"),
    ("ORRBCS", r"UNIFORM|\bUMP\b",                                 r"UNIFORM"),
    ("ORRBCS", r".",                                               r"COMMERCIAL"),
    # NM BCBS: Medicaid has no coded plan; Medicare Advantage vs Commercial
    ("NMBCBS", r"TURQUOISE|CENTENNIAL|MEDICAID",                   "NO_CODED_PLAN"),
    ("NMBCBS", r"MEDICARE|MED ADV|ADVANTAGE|DUAL",                 r"MEDICARE"),
    ("NMBCBS", r".",                                               r"COMMERCIAL"),
    # Horizon NJ: both coded plans are Medicare and share one list
    ("NJBCBS", r"MEDICARE|TOTALCARE|TOTAL CARE|ADVANTAGE|DUAL",    "ALL"),
    ("NJBCBS", r".",                                               "NO_CODED_PLAN"),
]


def resolve(payor, target):
    """Turn a target regex into the rules plan name(s) of that payor."""
    if target in ("ALL", "NO_CODED_PLAN"):
        return target
    hits = [p for p in coded_plans.get(payor, []) if re.search(target, p, re.I)]
    if not hits:
        print(f"  WARNING {payor}: no coded plan matches /{target}/ -> using ALL. "
              f"Coded plans: {coded_plans.get(payor)}")
        return "ALL"
    return "||".join(hits)


print("Resolved rules plans:")
resolved = []
for payor, pat, target in PLAN_RULES:
    plan = resolve(payor, target)
    resolved.append((payor, pat, plan))
    print(f"  {payor:<7} {pat:<50} -> {plan}")

# apply
codes = set(coded_plans)                    # payors that have at least one coded plan
all_codes = set(rules_df["PAYOR_CODE"].dropna().astype(str).str.strip())
pairs["rules_plan"] = ""
# coded list only: a rules payor with no CODED_LIST plan gets no list at all
pairs.loc[pairs["rules_payor"].isin(all_codes), "rules_plan"] = "NO_CODED_PLAN"
pairs.loc[pairs["rules_payor"].isin(codes), "rules_plan"] = "ALL"
both = (pairs["PAYOR"].astype(str) + " | " + pairs["PLAN"].astype(str)).str.upper()
for payor in dict.fromkeys(p for p, _, _ in resolved):
    todo = pairs["rules_payor"].eq(payor)
    for p, pat, plan in resolved:
        if p != payor:
            continue
        hit = todo & both.str.contains(pat, regex=True, na=False)
        pairs.loc[hit, "rules_plan"] = plan
        todo &= ~hit

# headline: which claims get a coded list (rules decide) vs not (model decides)
dd = pairs            # n is split by line of business, so sum all rows (no de-duplication here)
seg = pd.Series("NOT_A_RULES_PAYOR", index=dd.index)
seg[dd["rules_plan"].eq("NO_CODED_PLAN")] = "RULES_PAYOR_NO_CODED_LIST"
seg[dd["rules_plan"].ne("NO_CODED_PLAN") & dd["rules_plan"].ne("")] = "CODED_LIST"
tot = dd.groupby(seg)["n"].sum()
print("\nClaims by segment:")
print(pd.DataFrame({"claims": tot, "share": (tot / tot.sum()).round(3)}).to_string())

# summary for the four payors
show = pairs[pairs["rules_payor"].isin({p for p, _, _ in PLAN_RULES})]
print("\nClaims by rules payor x rules plan:")
print(show.groupby(["rules_payor", "rules_plan"])["n"].sum().to_string())

# rewrite master sheet with the new column next to rules_payor
master = pairs.copy()
cols = list(master.columns)
cols.insert(cols.index("rules_payor") + 1, cols.pop(cols.index("rules_plan")))
master = master[cols]
master["corrected_code"] = master.get("corrected_code", "")
master["cum_share"] = (master["n"].cumsum() / master["n"].sum()).round(3)
master.to_excel("payor_master_review.xlsx", index=False)
print("\nWrote payor_master_review.xlsx (now with rules_plan).")
