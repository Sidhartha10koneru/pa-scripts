"""
validate_mapping.py (v3) - checks every payor (and plan) mapping against the claims data.

Question, per claims payor: "Does the coded list we mapped it to separate its PA lines
from its non-PA lines?"

v3 changes:
  - lists built from CODED_LIST rows only, straight from rules_df (v2 used exploded_rules_v2,
    which may include non-coded rows and made lists too big)
  - plan level: uses rules_plan from plan_mapping.py; prints payor-level vs plan-level lift
  - optional PRIMARY_ONLY switch (FILING_ORDER_C == 1)

Needs: path, s3, rules_df, and `pairs` with rules_plan (run run_mapping.py, then plan_mapping.py).
Run:   %run -i validate_mapping.py
Output: mapping_validation.xlsx + printed summary.
"""
import re
import pandas as pd

MIN_TRUE_LINES = 20
FIT_LIFT, FIT_COV = 5, 0.25
WEAK_LIFT = 2
PRIMARY_ONLY = False          # set True once FILING_ORDER_C is confirmed (1 = primary payer)
COMPANION = r"^(99[2-4]\d\d|\d{4}F|36415|96[3-5]\d\d|Q996[5-7]|A957[5-9]|G2211)$"

if "pairs" not in globals() or "rules_plan" not in pairs:
    raise NameError("Run run_mapping.py and then plan_mapping.py first.")

# --------------------------------------------------------------- 1) coded lists
cr = rules_df[rules_df["CODING_STATUS"].astype(str).str.strip() == "CODED_LIST"].copy()
cr["PAYOR_CODE"] = cr["PAYOR_CODE"].astype(str).str.strip()
cr["PLAN_NAME"] = cr["PLAN_NAME"].astype(str).str.strip()
cr["cpt"] = (cr["PLAN_CPT_CODES"].astype(str).str.replace("_x000D_", ",", regex=False)
             .str.upper().str.split(r"[^0-9A-Z]+"))
cr = cr.explode("cpt")
cr = cr[cr["cpt"].str.fullmatch(r"[0-9A-Z]{5}", na=False)]
plan_list = cr.groupby(["PAYOR_CODE", "PLAN_NAME"])["cpt"].agg(set).to_dict()
payor_list = cr.groupby("PAYOR_CODE")["cpt"].agg(set).to_dict()


def list_for(payor, plan):
    if plan == "ALL":
        return payor_list.get(payor, set())
    if plan == "NO_CODED_PLAN" or not plan:
        return set()
    return set().union(*[plan_list.get((payor, p), set()) for p in plan.split("||")])


# --------------------------------------------------------------- 2) mapping + claims
all_codes = set(rules_df["PAYOR_CODE"].dropna().astype(str).str.strip())
mapping = pairs[["PAYOR", "PLAN", "rules_payor", "rules_plan"]].drop_duplicates(["PAYOR", "PLAN"])
mapping = mapping[mapping["rules_payor"].isin(all_codes)].copy()
lists = {(p, pl): list_for(p, pl) for p, pl in set(zip(mapping.rules_payor, mapping.rules_plan))}

print("Reading claims ...")
cols = ["CSN", "PAYOR", "PLAN", "FINAL_CPT", "GROUP_LABEL"] + (["FILING_ORDER_C"] if PRIMARY_ONLY else [])
c = pd.read_parquet(path, columns=cols, filesystem=s3)
c["GROUP_LABEL"] = c["GROUP_LABEL"].astype(str).str.upper()
c = c[c["GROUP_LABEL"].isin(["TRUE", "FALSE"])]
if PRIMARY_ONLY:
    c = c[pd.to_numeric(c["FILING_ORDER_C"], errors="coerce").eq(1)]
for k in ("PAYOR", "PLAN"):
    c[k] = c[k].astype(str)
c = c.merge(mapping, on=["PAYOR", "PLAN"], how="inner")
c["cpt"] = c["FINAL_CPT"].astype(str).str.strip().str.upper()
c["is_true"] = c["GROUP_LABEL"].eq("TRUE")
c["companion"] = c["cpt"].str.match(COMPANION)
c["on_own"] = [x in lists[(p, pl)] for p, pl, x in zip(c.rules_payor, c.rules_plan, c.cpt)]
c["on_payor"] = [x in payor_list.get(p, ()) for p, x in zip(c.rules_payor, c.cpt)]
print(f"Claim lines linked to a rules payor (TRUE/FALSE{', primary only' if PRIMARY_ONLY else ''}): {len(c):,}")


def lift(g, col):
    t, f = g.loc[g.is_true, col].mean(), g.loc[~g.is_true, col].mean()
    return t, f, t / max(f, 1 / max((~g.is_true).sum(), 1))


# --------------------------------------------------------------- 3) score
rows = []
for (payor, code, plan), a in c.groupby(["PAYOR", "rules_payor", "rules_plan"]):
    g = a[~a.companion]
    tc = a[a.is_true]
    size = len(lists[(code, plan)])
    row = {"claims_PAYOR": payor, "rules_payor": code, "rules_plan": plan,
           "claim_lines": len(a), "pa_lines": len(tc), "pa_claims": tc.CSN.nunique(),
           "own_list_size": size,
           "pa_claims_with_listed_line": round(a[a.CSN.isin(tc.CSN)].groupby("CSN").on_own.any().mean(), 3)
           if len(tc) else None}
    if size == 0:
        row["status"] = "NO_CODED_LIST"
    elif g.is_true.sum() < MIN_TRUE_LINES or (~g.is_true).sum() == 0:
        row["status"] = "NOT_TESTABLE"
    else:
        t, f, l = lift(g, "on_own")
        row.update(own_pa_cov=round(t, 3), own_nonpa_cov=round(f, 4), own_lift=round(l, 1),
                   payor_level_lift=round(lift(g, "on_payor")[2], 1))
        row["status"] = "FITS" if (l >= FIT_LIFT and t >= FIT_COV) else "WEAK_FIT" if l >= WEAK_LIFT else "DOES_NOT_FIT"
    rows.append(row)
res = pd.DataFrame(rows).sort_values("claim_lines", ascending=False)

# --------------------------------------------------------------- 4) summary
vol = res.groupby("status")["claim_lines"].sum()
print("\nClaim lines by status:")
print((vol / vol.sum() * 100).round(1).astype(str).add("%").to_string())
show = ["claims_PAYOR", "rules_payor", "rules_plan", "claim_lines", "pa_lines", "own_list_size",
        "own_pa_cov", "own_lift", "payor_level_lift", "status"]
print("\nTop 25 by volume:")
print(res[[k for k in show if k in res]].head(25).to_string(index=False))
print("\nPlan-level payors (payor_level_lift = old way, own_lift = new way):")
print(res[res.rules_plan.ne("ALL")][[k for k in show if k in res]].head(25).to_string(index=False))

with pd.ExcelWriter("mapping_validation.xlsx") as w:
    res.to_excel(w, sheet_name="by_payor_plan", index=False)
    pd.DataFrame({"setting": ["lists", "MIN_TRUE_LINES", "FITS", "WEAK_FIT", "COMPANION excluded", "PRIMARY_ONLY"],
                  "value": ["CODED_LIST rows of rules_df, per rules plan", MIN_TRUE_LINES,
                            f"lift >= {FIT_LIFT} and PA coverage >= {FIT_COV}", f"lift >= {WEAK_LIFT}",
                            COMPANION, PRIMARY_ONLY]}).to_excel(w, sheet_name="method", index=False)
print("\nWrote mapping_validation.xlsx")
