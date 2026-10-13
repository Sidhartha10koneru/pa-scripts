"""
run_mapping.py - builds the payor master sheet in one go.

Needs (already in your notebook): df_cols (claims) and rules_df (rules).
Run in a notebook cell:   %run -i run_mapping.py

Output: payor_mapping_review.xlsx  (one row per claims PAYOR/PLAN, biggest first)
"""
import os
import pandas as pd

# load the helper functions (file was saved as Payor_tag or payor_tagging.py)
for f in ("payor_tagging.py", "Payor_tag"):
    if os.path.exists(f):
        exec(open(f).read())
        break
else:
    raise FileNotFoundError("Download Payor_tag first (the helper script).")

# ---------------------------------------------------------------------------
# Name pattern  ->  rules PAYOR_CODE.   Checked top to bottom, first hit wins.
# The text searched is "PAYOR | PLAN" in capitals.
# Edit codes here if the check below says one doesn't exist in the rules.
# ---------------------------------------------------------------------------
KEYWORDS = [
    # traditional Medicare: payor is just "MEDICARE" (no company) -> CMS
    (r"^MEDICARE \|",                      "CMSHSP"),
    # names that look like something else
    (r"EMPIRE PLAN",                       "NOT_IN_RULES"),   # NY state employees, not Empire BCBS
    (r"INLAND EMPIRE",                     "IEHP"),
    (r"\bUMR\b",                           "NOT_IN_RULES"),   # UHC admin for employer plans - review later
    (r"GLOBE LIFE|BANKERS|NORTHWELL|CULINARY|TRANSAMERICA|TRILLIUM", "NOT_IN_RULES"),
    # Medigap / supplement plans pay after Medicare and don't run PA
    (r"SUPPLEMENT|MEDIGAP",                "SUPPLEMENT_NO_PA"),
    # state Medicaid and state-named Blues
    (r"MEDICAID - COLORADO",               "COCAID"),
    (r"BLUE SHIELD OF CALIFORNIA",         "CABS"),
    (r"BLUE CROSS OF CALIFORNIA",          "CABCBS"),
    (r"BCBS OF ARIZONA",                   "AZBCBS"),
    (r"BCBS OF NM|BCBS NM|TURQUOISE",      "NMBCBS"),
    (r"TRICARE WEST",                      "TRICRW"),
    # specific brands before their parent company
    (r"MERITAIN",                          "MERITN"),     # Aetna-owned admin; must come before AETNA
    (r"OXFORD",                            "OXFORD"),
    (r"\bAARP\b",                          "AARP"),
    (r"BCBS OF OREGON|REGENCE",            "ORRBCS"),
    (r"HORIZON NJ HEALTH",                 "NJENHD"),
    (r"HORIZON",                           "NJBCBS"),
    (r"ANTHEM[^|]*MEDI(CARE|BLUE)",        "MEDADV"),
    (r"ANTHEM BCBS CO\b",                  "COBCBS"),
    (r"\bGHI\b",                           "GHINY"),
    (r"\bHIP\b",                           "HIPNY"),
    (r"HEALTHFIRST",                       "NYHFST"),
    (r"FIDELIS",                           "FDCRNY"),
    (r"CDPHP|CAPITAL DISTRICT",            "CDPHP"),
    # one company -> one code
    (r"UNITED ?HEALTH ?CARE|\bUHC\b",      "UHC"),
    (r"\bAETNA\b",                         "AETNA"),
    (r"\bCIGNA\b",                         "CIGNA"),
    (r"\bMVP\b",                           "NYMVP"),
    (r"\b1199\b",                          "1199NB"),
    (r"PACIFICSOURCE",                     "PSHLTH"),
    (r"MAGNACARE",                         "MAGNA"),
    (r"HUMANA",                            "HUMANU"),
    (r"PROVIDENCE",                        "PROVHP"),
    (r"\bMODA\b",                          "MDAHLT"),
    (r"PRESBYTERIAN",                      "PRESBY"),
    (r"CHAMPVA",                           "VETHAC"),
    (r"MUTUAL OF OMAHA",                   "MUTOMH"),
]

# 1) check every code really exists in the rules file
codes = set(rules_df["PAYOR_CODE"].dropna().astype(str).str.strip())
SPECIAL = {"NOT_IN_RULES", "SUPPLEMENT_NO_PA"}
bad = sorted({c for _, c in KEYWORDS if c not in SPECIAL and c not in codes})
if bad:
    print("WARNING - these codes are not in rules_df, fix them in KEYWORDS:", bad)
else:
    print("All codes found in the rules file.")

# 2) save as keywords.xlsx and apply to every claims PAYOR/PLAN pair
pd.DataFrame([{"pattern": p, "tag": c, "priority": i, "note": ""}
              for i, (p, c) in enumerate(KEYWORDS)]).to_excel("keywords.xlsx", index=False)
pairs = apply_keywords(df_cols, "keywords.xlsx")

# supplement plans: the word is often in PLAN while PAYOR says "UNITED HEALTHCARE",
# so override whatever the payor matched
sup = pairs["lob"].eq("SUPPLEMENT")
pairs.loc[sup, "rules_payor"] = "SUPPLEMENT_NO_PA"
print(f"Supplement rows set aside: {pairs.loc[sup, 'n'].sum():,} claims")

# 3) master sheet for review: biggest names first, with a column for your corrections
master = pairs.copy()
master["corrected_code"] = ""
master["cum_share"] = (master["n"].cumsum() / master["n"].sum()).round(3)
master.to_excel("payor_master_review.xlsx", index=False)
print("\nWrote payor_master_review.xlsx - open it and check the top rows.")
