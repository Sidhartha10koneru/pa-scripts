"""
validate_mapping.py - checks every payor mapping in the master sheet against the claims data.

Question it answers, per mapped payor:
  "Does the rules list we mapped this payor to fit its PA claims better than
   any other payor's list in the rules file?"

Needs in the notebook: path, s3 (claims parquet), rules_df, exploded_rules_v2.
Run AFTER run_mapping.py in the same kernel (it uses `pairs`), or it runs it for you:
    %run -i validate_mapping.py

Output: mapping_validation.xlsx  +  a printed summary.
"""
import os
import re
import pandas as pd

MIN_TRUE_LINES = 20          # fewer PA lines than this -> NOT_TESTABLE
NEAR_BEST = 0.90             # mapped code within 90% of the best score -> CONFIRMED
# Lines that ride along on a PA claim but never need PA themselves.
# Excluded from scoring so they don't blur the comparison.
COMPANION = r"^(99[2-4]\d\d|\d{4}F|36415|96[3-5]\d\d|Q996[5-7]|A957[5-9]|G2211)$"

# --------------------------------------------------------------- 1) mapping
if "pairs" not in globals():
    print("`pairs` not found - running run_mapping.py first ...")
    exec(open("run_mapping.py").read())
mapping = pairs[["PAYOR", "PLAN", "rules_payor"]].copy()
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
    if t.sum() < MIN_TRUE_LINES or f.sum() == 0:
        row["status"] = "NOT_TESTABLE"
        rows.append(row)
        continue
    cov_t = r["cpt_code"].map(t).fillna(0).groupby(r["PAYOR_CODE"]).sum() / t.sum()
    cov_f = r["cpt_code"].map(f).fillna(0).groupby(r["PAYOR_CODE"]).sum() / f.sum()
    # score = PA coverage minus non-PA coverage (Youden's J):
    # tiny lists score low (little coverage), huge lists score low (flag everything)
    score = (cov_t - cov_f).sort_values(ascending=False)
    own_score = score.get(code, 0.0)
    best = score.index[0]
    row.update({"own_pa_cov": round(cov_t.get(code, 0), 3),
                "own_nonpa_cov": round(cov_f.get(code, 0), 4),
                "own_lift": round(cov_t.get(code, 0) / max(cov_f.get(code, 0), 1 / f.sum()), 1),  # floor = one non-PA line
                "own_score": round(own_score, 3),
                "own_rank_of_139": int(list(score.index).index(code) + 1) if code in score.index else None,
                "best_code": best,
                "best_score": round(score.iloc[0], 3),
                "runner_up": score.index[1] if score.index[0] == code else best})
    if own_score <= 0:
        row["status"] = f"REVIEW - list does not fit (best: {best})"
    elif best == code or own_score >= NEAR_BEST * score.iloc[0]:
        row["status"] = "CONFIRMED"
    else:
        row["status"] = f"REVIEW - {best} fits better"
    rows.append(row)

res = pd.DataFrame(rows).sort_values("claim_lines", ascending=False)

# --------------------------------------------------------------- 5) summary
vol = res.groupby(res["status"].str.split(" - ").str[0])["claim_lines"].sum()
print("\nClaim lines by status:")
print((vol / vol.sum() * 100).round(1).astype(str).add("%").to_string())
cols = ["claims_PAYOR", "rules_payor", "claim_lines", "pa_lines", "own_pa_cov", "own_lift",
        "own_rank_of_139", "best_code", "pa_claims_with_listed_line", "status"]
print("\nTop 25 mapped payors by volume:")
print(res[[k for k in cols if k in res.columns]].head(25).to_string(index=False))

with pd.ExcelWriter("mapping_validation.xlsx") as w:
    res.to_excel(w, sheet_name="by_payor", index=False)
    pd.DataFrame({"setting": ["MIN_TRUE_LINES", "NEAR_BEST", "COMPANION codes excluded", "score"],
                  "value": [MIN_TRUE_LINES, NEAR_BEST, COMPANION,
                            "PA-line coverage minus non-PA-line coverage (Youden J), ranked over all rules payors"]}
                 ).to_excel(w, sheet_name="method", index=False)
print("\nWrote mapping_validation.xlsx")
