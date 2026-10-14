"""
rule_conditions.py - read the diagnosis conditions out of the rules' "Auth Requirement" text
and test whether CPT + diagnosis fits the claims better than CPT alone.

Needs in the notebook: path, s3, rules_df (sheet "Plans with Services"), and `pairs` from run_mapping.py.
Run:   %run -i rule_conditions.py
Outputs:
  rule_conditions.xlsx  - one row per coded rule row with its parsed condition (for review)
  printed comparison    - CPT-only vs CPT+diagnosis fit, per payor
"""
import re
import numpy as np
import pandas as pd

AUTH_COL = [c for c in rules_df.columns if "Auth Requirement" in c][0]
ICD = r"[A-TV-Z]\d[0-9A-Z](?:\.?[0-9A-Z]{1,4})?"
CANCER = [("C00", "D49")]          # ICD-10 neoplasm chapter, used for "cancer/oncology diagnosis"


def norm_dx(s):
    return re.sub(r"[^0-9A-Z]", "", str(s).upper())


def parse(text):
    """Return dict: dx_mode ('include' / 'exclude' / None), dx_codes, dx_ranges, flags."""
    t = " ".join(str(text).split())
    low = t.lower()
    out = {"dx_mode": None, "dx_codes": [], "dx_ranges": [], "code_specific": False,
           "outpatient_only": "outpatient setting" in low or "outpatient only" in low,
           "inpatient_only": "inpatient only" in low,
           "cost_threshold": bool(re.search(r"\$\s?\d", t)),
           "vendor": "evicore" in low or "optum" in low}

    # ranges like "C34.00 - C34.92"
    for a, b in re.findall(rf"({ICD})\s*[-–]\s*({ICD})", t):
        out["dx_ranges"].append((norm_dx(a), norm_dx(b)))
    t_wo_ranges = re.sub(rf"{ICD}\s*[-–]\s*{ICD}", " ", t)
    # single codes, only from the part of the text that talks about diagnoses
    m = re.search(r"diagnos\w*|\bdx\b", t_wo_ranges, re.I)
    if m:
        out["dx_codes"] = [norm_dx(c) for c in re.findall(rf"\b{ICD}\b", t_wo_ranges[m.start():])]
    if re.search(r"cancer diagnosis|oncology diagnosis", low):
        out["dx_ranges"] += CANCER

    has_dx = bool(out["dx_codes"] or out["dx_ranges"])
    if has_dx:
        if re.search(r"not required (when billed with|for) the following|not required for the\s+following diagnos", low):
            out["dx_mode"] = "exclude"
        elif re.search(r"only requires|only when|only require|when submitted with|with the following|"
                       r"for (this|these) diagnos|for a cancer diagnosis|oncology diagnosis|"
                       r"required with the following|dx codes", low):
            out["dx_mode"] = "include"
    # "Code 38232 will only require..." -> condition applies to named codes only: flag, don't apply
    if re.search(r"\bcodes?\s+[0-9A-Z]{5}\b.{0,40}\bonly\b", t, re.I) or \
       re.search(r"\bcodes?\s+([0-9A-Z]{5},?\s*){2,}.{0,30}also require", t, re.I):
        out["code_specific"] = True
    return out


def dx_match(dx, codes, ranges):
    if not dx:
        return False
    for c in codes:                       # prefix match: E11 matches E1165
        if dx.startswith(c) or c.startswith(dx):
            return True
    for a, b in ranges:                   # inclusive range on normalised codes
        if a <= dx[:len(a)] and dx[:len(b)] <= b:
            return True
    return False


# ------------------------------------------------------------------ 1) parse coded rule rows
coded = rules_df[rules_df["CODING_STATUS"] == "CODED_LIST"].copy()
parsed = coded[AUTH_COL].fillna("").map(parse).apply(pd.Series)
coded = pd.concat([coded.reset_index(drop=True), parsed.reset_index(drop=True)], axis=1)
coded["applies_dx"] = coded["dx_mode"].notna() & ~coded["code_specific"]

print("Coded rule rows:", len(coded))
print("  with a usable diagnosis condition:", int(coded.applies_dx.sum()),
      f"({coded.applies_dx.mean():.1%})  include={int((coded.dx_mode=='include').sum())}"
      f"  exclude={int((coded.dx_mode=='exclude').sum())}")
print("  code-specific (flagged, not applied):", int(coded.code_specific.sum()))
coded.assign(dx_codes=coded.dx_codes.map(" ".join),
             dx_ranges=coded.dx_ranges.map(lambda r: " ".join(f"{a}-{b}" for a, b in r))
             )[["PAYOR_CODE", "PLAN_NAME", "SERVICE_DESCRIPTION", AUTH_COL, "dx_mode", "dx_codes",
                "dx_ranges", "code_specific", "outpatient_only", "inpatient_only", "cost_threshold",
                "vendor"]].to_excel("rule_conditions.xlsx", index=False)

# ------------------------------------------------------------------ 2) explode to payor x CPT
coded["cpt_code"] = (coded["PLAN_CPT_CODES"].astype(str).str.replace("_x000D_", ",", regex=False)
                     .str.upper().str.split(r"[^0-9A-Z]+"))
ex = coded.explode("cpt_code").reset_index(drop=True)
ex["PAYOR_CODE"] = ex["PAYOR_CODE"].astype(str)
ex = ex[ex["cpt_code"].str.fullmatch(r"[0-9A-Z]{5}", na=False)]
uncond = set(zip(ex.loc[~ex.applies_dx, "PAYOR_CODE"].astype(str), ex.loc[~ex.applies_dx, "cpt_code"]))
cond = (ex[ex.applies_dx].groupby(["PAYOR_CODE", "cpt_code"])
        [["dx_mode", "dx_codes", "dx_ranges"]].apply(lambda g: list(zip(g.dx_mode, g.dx_codes, g.dx_ranges)))
        .to_dict())
listed = uncond | set(cond)

# ------------------------------------------------------------------ 3) claims
print("Reading claims ...")
c = pd.read_parquet(path, columns=["PAYOR", "PLAN", "FINAL_CPT", "Primary_DX", "GROUP_LABEL"], filesystem=s3)
c = c[c["GROUP_LABEL"].astype(str).isin(["TRUE", "FALSE"])]
for k in ("PAYOR", "PLAN"):
    c[k] = c[k].astype(str)
c = c.merge(pairs[["PAYOR", "PLAN", "rules_payor"]].drop_duplicates(["PAYOR", "PLAN"]), on=["PAYOR", "PLAN"])
c["cpt"] = c["FINAL_CPT"].astype(str).str.strip().str.upper()
c["dx"] = c["Primary_DX"].map(norm_dx)
c["pa"] = c["GROUP_LABEL"].astype(str).eq("TRUE")
key = list(zip(c["rules_payor"].astype(str), c["cpt"]))
c["on_list"] = [k in listed for k in key]


def rule_with_dx(k, dx):
    if k in uncond:
        return True
    for mode, codes, ranges in cond.get(k, []):
        hit = dx_match(dx, codes, ranges)
        if (mode == "include" and hit) or (mode == "exclude" and not hit):
            return True
    return False


sub = c["on_list"]
c["rule_dx"] = False
c.loc[sub, "rule_dx"] = [rule_with_dx(k, d) for k, d, s in zip(key, c["dx"], sub) if s]
c["conditional"] = [k in cond for k in key]

# ------------------------------------------------------------------ 4) compare
def fit(g, col):
    t, f = g.loc[g.pa, col].mean(), g.loc[~g.pa, col].mean()
    return pd.Series({"pa_cov": t, "nonpa_cov": f, "lift": t / max(f, 1 / max((~g.pa).sum(), 1))})

rows = []
for p, g in c.groupby("rules_payor"):
    if g.pa.sum() < 20 or not g.on_list.any():
        continue
    a, b = fit(g, "on_list"), fit(g, "rule_dx")
    rows.append({"rules_payor": p, "lines": len(g), "pa_lines": int(g.pa.sum()),
                 "listed_lines_conditional": round(g.loc[g.on_list, "conditional"].mean(), 3),
                 "cpt_only_pa_cov": round(a.pa_cov, 3), "cpt_only_lift": round(a.lift, 1),
                 "cpt_dx_pa_cov": round(b.pa_cov, 3), "cpt_dx_lift": round(b.lift, 1)})
res = pd.DataFrame(rows).sort_values("lines", ascending=False)
print("\nListed claim lines under a conditional rule:", f"{c.loc[c.on_list, 'conditional'].mean():.1%}")
print(res.head(30).to_string(index=False))
res.to_excel("rule_conditions_fit.xlsx", index=False)
print("\nWrote rule_conditions.xlsx and rule_conditions_fit.xlsx")
