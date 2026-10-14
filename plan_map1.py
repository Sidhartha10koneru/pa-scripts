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
    (r"EMPIRE PLAN",                       "NYSHIP"),         # NY state employees (NYSHIP), not Empire BCBS
    (r"MEDICAID - NEW YORK",               "NYCAID"),
    (r"MEDICAID - NEVADA",                 "CAIDNV"),
    (r"METROPLUS",                         "METPL"),
    (r"OSCAR",                             "NYOSCR"),         # claims are mostly NY; NJ would be NJOSCR
    (r"WELLCARE",                          "WLCRNY"),
    (r"MOLINA|AFFINITY",                   "NOT_IN_RULES"),   # rules have no Molina New York
    (r"INLAND EMPIRE",                     "IEHP"),
    (r"\bUMR\b",                           "NOT_IN_RULES"),   # UHC admin for employer plans - review later
    (r"GLOBE LIFE|BANKERS|NORTHWELL|CULINARY|TRANSAMERICA|TRILLIUM", "NOT_IN_RULES"),
    (r"GENERIC|OMC TPA|ONE CALL",          "NOT_IN_RULES"),   # placeholder / admin / workers' comp
    (r"KAISER PERMANENTE[^|]*COLORADO|KAISER[^|]*\|[^|]*COLORADO", "COKAIS"),
    (r"TRICARE FOR LIFE",                  "TRCRU"),
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
    (r"\bAARP\b",                          "UHC"),        # AARP plans are insured by UHC; UHC list fits AARP PA claims best (lift 8.8)
    (r"BCBS OF OREGON|REGENCE",            "ORRBCS"),
    (r"HORIZON NJ HEALTH",                 "NJENHD"),
    (r"HORIZON",                           "NJBCBS"),
    (r"ANTHEM[^|]*MEDI(CARE|BLUE)",        "MEDADV"),
    (r"ANTHEM BCBS CO\b",                  "COBCBS"),
    (r"^ANTHEM",                           "EMHECH"),     # patient ZIPs show these are NY -> Anthem BCBS of New York
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

# ---------------------------------------------------------------------------
# OVERRIDES: applied last, on "PAYOR | PLAN" together, and they always win.
# Used when the PLAN name says something the PAYOR name hides.
# ---------------------------------------------------------------------------
OVERRIDES = [
    (r"SUPPLEMENT|MEDIGAP",                "SUPPLEMENT_NO_PA"),  # pays after Medicare, no PA
    (r"OPTUM CARE NETWORK",                "DELEGATED"),         # delegated medical group: its own auth rules
    (r"BLUE ?CARD|AHN ANTHEM",             "BLUECARD"),          # out-of-area Blue member: home plan rules apply
    (r"FEDERAL",                           "FEP"),               # Federal Employee Program: national rules
    (r"UNITED ?HEALTH ?CARE MEDICA(?:ID)?\b|\bUHC MEDICAID|UNITED ?HEALTH ?CARE COMMUNITY PLAN", "UHCNY"),  # UHC NY Medicaid (MEDICA\b = truncated name, not MEDICARE)
    (r"PRESBYTERIAN",                      "PRESBY"),            # Presbyterian Turquoise is Presbyterian's plan, not BCBS NM
    (r"BCBS MED ADV DUAL|BCBS MEDICARE PPO|CENTENNIAL BCBS|BC/BS UNM|TURQUOISE", "NMBCBS"),  # NM ZIPs
    (r"ANTHEM.*MEDI(?:CARE|BLUE)",         "MEDADV"),            # Anthem Medicare Advantage
]

# 1) check every code really exists in the rules file
codes = set(rules_df["PAYOR_CODE"].dropna().astype(str).str.strip())
SPECIAL = {"NOT_IN_RULES", "SUPPLEMENT_NO_PA", "BLUECARD", "FEP", "DELEGATED"}
bad = sorted({c for _, c in KEYWORDS + OVERRIDES if c not in SPECIAL and c not in codes})
if bad:
    print("WARNING - these codes are not in rules_df, fix them in KEYWORDS:", bad)
else:
    print("All codes found in the rules file.")

# 2) save as keywords.xlsx and apply to every claims PAYOR/PLAN pair
pd.DataFrame([{"pattern": p, "tag": c, "priority": i, "note": ""}
              for i, (p, c) in enumerate(KEYWORDS)]).to_excel("keywords.xlsx", index=False)
pairs = apply_keywords(df_cols, "keywords.xlsx")

# apply OVERRIDES (first match wins, and it replaces the earlier result)
import re
both = (pairs["PAYOR"].astype(str) + " | " + pairs["PLAN"].astype(str)).str.upper()
done = pd.Series(False, index=pairs.index)
for pat, code in OVERRIDES:
    hit = both.str.contains(pat, regex=True) & ~done
    pairs.loc[hit, ["rules_payor", "matched_pattern", "matched_on"]] = [code, pat, "OVERRIDE"]
    done |= hit
    print(f"Override {code:<17} {pairs.loc[hit, 'n'].sum():>10,} claims")

# 2b) the honest number: claims linked to a real rules payor code
real = pairs["rules_payor"].isin(codes)
print(f"\nClaims linked to a real rules payor: {pairs.loc[real, 'n'].sum() / pairs['n'].sum():.1%}")
print(pairs.groupby(pairs["rules_payor"].where(~real, "LINKED_TO_RULES").fillna("UNMATCHED"))["n"]
      .sum().sort_values(ascending=False).to_string())

# 3) master sheet for review: biggest names first, with a column for your corrections
master = pairs.copy()
master["corrected_code"] = ""
master["cum_share"] = (master["n"].cumsum() / master["n"].sum()).round(3)
master.to_excel("payor_master_review.xlsx", index=False)
print("\nWrote payor_master_review.xlsx - open it and check the top rows.")
