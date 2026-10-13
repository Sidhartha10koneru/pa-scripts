"""
validate_mapping.py - checks every payor mapping in the master sheet against the claims data.

Question it answers, per claims payor:
  "Does the coded list of the rules payor we mapped it to actually separate
   its PA lines from its non-PA lines?"

v2 changes:
  - fixed duplicate rows (mapping had one row per LINE_OF_BUSINESS -> lines were counted several times)
  - payors whose rules code has no coded CPT list -> NO_CODED_LIST (was wrongly "does not fit")
  - status now judged on the payor's OWN list (lift + coverage). "best_code" is kept for
    information only: big lists from other states win that ranking simply by being big.

Needs in the notebook: path, s3 (claims parquet), rules_df, exploded_rules_v2.
Run AFTER run_mapping.py in the same kernel (it uses `pairs`), or it runs it for you:
    %run -i validate_mapping.py

Output: mapping_validation.xlsx  +  a printed summary.
"""
import os
import re
import pandas as pd

MIN_TRUE_LINES = 20          # fewer PA lines than this -> NOT_TESTABLE
FIT_LIFT, FIT_COV = 5, 0.25  # FITS: own list >=5x more common on PA lines and covers >=25% of them
WEAK_LIFT = 2                # WEAK_FIT: lift between 2 and 5
# Lines that ride along on a PA claim but never need PA themselves.
# Excluded from scoring so they don't blur the comparison.
COMPANION = r"^(99[2-4]\d\d|\d{4}F|36415|96[3-5]\d\d|Q996[5-7]|A957[5-9]|G2211)$"

# --------------------------------------------------------------- 1) mapping
if "pairs" not in globals():
    print("`pairs` not found - running run_mapping.py first ...")
    exec(open("run_mapping.py").read())
mapping = pairs[["PAYOR", "PLAN", "rules_payor"]].drop_duplicates(["PAYOR", "PLAN"]).copy()
codes = set(rules_df["PAYOR_CODE"].dropna().astype(str).str.strip())
mapping = mapping[mapping["rules_payor"].isin(codes)]

# --------------------------------------------------------------- 2) claims
print("Reading claims ...")
c = pd.read_parquet(path, columns=["CSN", "PAYOR", "PLAN", "FINAL_CPT", "GROUP_LABEL"], filesystem=s3)
c["GROUP_LABEL"] = c["GROUP_LABEL"].astype(str).str.upper()
c = c[c["GROUP_LABEL"].isin(["TRUE", "FALSE"])]
for col in ("PAYOR", "PLAN"):
    c[col] = c[col].astype(str)
c = c.merge(mapping, on=["PAYOR", "PLAN"], how="inner")
c["cpt"] = c["FINAL_CPT"].astype(str).str.strip().str.upper()
c["is_true"] = c["GROUP_LABEL"].eq("TRUE")
c["companion"] = c["cpt"].str.match(COMPANION)
print(f"Claim lines linked to a rules payor (TRUE/FALSE only): {len(c):,}")

# --------------------------------------------------------------- 3) rules lists
r = (exploded_rules_v2[["PAYOR_CODE", "cpt_code"]].dropna().astype(str)
     .drop_duplicates())
r["cpt_code"] = r["cpt_code"].str.strip().str.upper()
own = set(zip(r["PAYOR_CODE"], r["cpt_code"]))
list_size = r.groupby("PAYOR_CODE").size()

# claim-level: does the PA claim have at least one line on its own payor's list?
c["on_own"] = [(p, x) in own for p, x in zip(c["rules_payor"], c["cpt"])]

# --------------------------------------------------------------- 4) score every claims payor
# Scored per claims PAYOR (not per rules code), so one wrong name hidden among
# correct ones under the same code still shows up.
rows = []
scored = c[~c["companion"]]
for (payor, code), g in scored.groupby(["PAYOR", "rules_payor"]):
    t = g.loc[g["is_true"], "cpt"].value_counts()
    f = g.loc[~g["is_true"], "cpt"].value_counts()
    all_lines = c[(c["PAYOR"] == payor) & (c["rules_payor"] == code)]
    tc = all_lines.loc[all_lines["is_true"]]
    row = {"claims_PAYOR": payor, "rules_payor": code,
           "claim_lines": len(all_lines),
           "pa_lines": int(tc.shape[0]),
           "pa_lines_scored": int(t.sum()),
           "pa_claims": tc["CSN"].nunique(),
           "pa_claims_with_listed_line": round(
               all_lines[all_lines["CSN"].isin(tc["CSN"])].groupby("CSN")["on_own"].any().mean(), 3)
               if len(tc) else None}
    row["own_list_size"] = int(list_size.get(code, 0))
    if row["own_list_size"] == 0:
        row["status"] = "NO_CODED_LIST"
        rows.append(row)
        continue
    if t.sum() < MIN_TRUE_LINES or f.sum() == 0:
        row["status"] = "NOT_TESTABLE"
        rows.append(row)
        continue
    cov_t = r["cpt_code"].map(t).fillna(0).groupby(r["PAYOR_CODE"]).sum() / t.sum()
    cov_f = r["cpt_code"].map(f).fillna(0).groupby(r["PAYOR_CODE"]).sum() / f.sum()
    score = (cov_t - cov_f).sort_values(ascending=False)
    best = score.index[0]
    pa_cov = cov_t.get(code, 0)
    lift = pa_cov / max(cov_f.get(code, 0), 1 / f.sum())   # floor = one non-PA line
    row.update({"own_pa_cov": round(pa_cov, 3),
                "own_nonpa_cov": round(cov_f.get(code, 0), 4),
                "own_lift": round(lift, 1),
                "info_best_code": best,
                "info_best_list_size": int(list_size.get(best, 0)),
                "info_own_rank": int(list(score.index).index(code) + 1)})
    if lift >= FIT_LIFT and pa_cov >= FIT_COV:
        row["status"] = "FITS"
    elif lift >= WEAK_LIFT:
        row["status"] = "WEAK_FIT"
    else:
        row["status"] = "DOES_NOT_FIT"
    rows.append(row)

res = pd.DataFrame(rows).sort_values("claim_lines", ascending=False)

# --------------------------------------------------------------- 5) summary
vol = res.groupby("status")["claim_lines"].sum()
print("\nClaim lines by status:")
print((vol / vol.sum() * 100).round(1).astype(str).add("%").to_string())
cols = ["claims_PAYOR", "rules_payor", "claim_lines", "pa_lines", "own_list_size", "own_pa_cov",
        "own_lift", "pa_claims_with_listed_line", "status"]
print("\nTop 25 mapped payors by volume:")
print(res[[k for k in cols if k in res.columns]].head(25).to_string(index=False))

with pd.ExcelWriter("mapping_validation.xlsx") as w:
    res.to_excel(w, sheet_name="by_payor", index=False)
    pd.DataFrame({"setting": ["MIN_TRUE_LINES", "FITS", "WEAK_FIT", "COMPANION codes excluded", "info_best_code"],
                  "value": [MIN_TRUE_LINES, f"lift >= {FIT_LIFT} and PA coverage >= {FIT_COV}",
                            f"lift >= {WEAK_LIFT}", COMPANION,
                            "information only - rules payor whose list scores highest (PA cov minus non-PA cov); favours large lists"]}
                 ).to_excel(w, sheet_name="method", index=False)
print("\nWrote mapping_validation.xlsx")
