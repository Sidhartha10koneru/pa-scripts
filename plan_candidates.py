"""
plan_candidates.py - step 1 of claims -> rules PLAN mapping.

  a) sizes the problem: which rules payor codes (XREF_ID) have 1 plan vs several,
     and how many claims sit behind each (via the master sheet's rules_payor)
  b) single-plan payor codes: plan is decided automatically
  c) multi-plan payor codes: for every claims PAYOR|PLAN, proposes the top 3 rules plans
     (name similarity + segment hint) for a person to approve

Run:
  python plan_candidates.py --plans rules_analysis/plan_summary.csv \
                            --master payor_master_review.xlsx --out plan_candidates.xlsx
Master sheet needs columns: PAYOR, PLAN, n (claims), rules_payor.
"""
import argparse
import re
from difflib import SequenceMatcher

import pandas as pd

try:
    from rapidfuzz import fuzz
    sim = lambda a, b: fuzz.token_set_ratio(a, b) / 100
except ImportError:
    def sim(a, b):
        ta, tb = set(a.split()), set(b.split())
        tok = len(ta & tb) / max(1, min(len(ta), len(tb)))
        return max(tok, SequenceMatcher(None, a, b).ratio())

SEGMENTS = [  # first hit wins; same tag on both sides = boost
    ("MEDICARE", r"MEDICARE|\bMA\b|ADVANTAGE|AARP|MAPD|\bDSNP\b|D-SNP|DUAL|SENIOR|MEDIBLUE|COMPLETE"),
    ("MEDICAID", r"MEDICAID|COMMUNITY PLAN|\bCHIP\b|\bSTAR\b|CAID|MANAGED LONG|\bHARP\b|TURQUOISE|CENTENNIAL"),
    ("FEP",      r"FEDERAL|\bFEP\b"),
    ("EXCHANGE", r"EXCHANGE|MARKETPLACE|\bQHP\b|ESSENTIAL|INDIVIDUAL"),
    ("TRICARE",  r"TRICARE|CHAMPVA"),
]
STOP = r"\b(INC|CO|COMPANY|OF|THE|PLAN|PLANS|HEALTH|INSURANCE|INS|CORP|LLC)\b"


def norm(s):
    s = re.sub(r"[^A-Z0-9 ]", " ", str(s).upper())
    return re.sub(r"\s+", " ", re.sub(STOP, " ", s)).strip()


def segment(s):
    s = str(s).upper()
    return next((tag for tag, pat in SEGMENTS if re.search(pat, s)), "OTHER")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", default="rules_analysis/plan_summary.csv")
    ap.add_argument("--master", default="payor_master_review.xlsx")
    ap.add_argument("--out", default="plan_candidates.xlsx")
    a = ap.parse_args()

    plans = pd.read_csv(a.plans).rename(columns={"file_profile": "PLAN_ID"})
    plans["label"] = plans.payer_name.fillna("") + " " + plans.plan_name.fillna("")
    plans["seg"] = plans.label.map(segment)
    n_plans = plans.groupby("xref_id").PLAN_ID.nunique().rename("n_plans")

    m = pd.read_excel(a.master)
    m["rules_payor"] = m["rules_payor"].astype(str).str.strip()
    m = m.join(n_plans, on="rules_payor")
    m["case"] = pd.cut(m.n_plans.fillna(0), [-1, 0, 1, 1e9], labels=["no rules plan", "single plan", "multi plan"])

    # ---- a) sizing
    tot = m.n.sum()
    size = m.groupby("case", observed=True).agg(claim_names=("n", "size"), claims=("n", "sum"),
                                                payor_codes=("rules_payor", "nunique"))
    size["share_claims"] = (size.claims / tot).round(3)
    print("\n=== Sizing ===\n", size.to_string())
    print("\nRules payor codes by number of plans:\n", n_plans.value_counts().sort_index().to_string())
    multi = (m[m.case == "multi plan"].groupby("rules_payor")
             .agg(claims=("n", "sum"), claim_names=("n", "size"), n_plans=("n_plans", "first"))
             .sort_values("claims", ascending=False))
    multi["cum_share_of_multi"] = (multi.claims.cumsum() / multi.claims.sum()).round(3)
    print("\nMulti-plan payor codes, biggest first:\n", multi.head(25).to_string())

    # ---- b) single plan: automatic
    single = m[m.case == "single plan"].merge(
        plans[["xref_id", "PLAN_ID", "plan_name"]], left_on="rules_payor", right_on="xref_id", how="left")

    # ---- c) multi plan: top 3 candidates per claims name
    rows = []
    for _, r in m[m.case == "multi plan"].iterrows():
        text = f"{r.PAYOR} {r.PLAN}"
        cseg, ctext = segment(text), norm(text)
        cand = plans[plans.xref_id == r.rules_payor].copy()
        cand["score"] = [sim(ctext, norm(t)) + (0.25 if s == cseg and s != "OTHER" else 0)
                         - (0.25 if cseg != "OTHER" and s not in (cseg, "OTHER") else 0)
                         for t, s in zip(cand.label, cand.seg)]
        cand = cand.nlargest(3, "score")
        out = {"PAYOR": r.PAYOR, "PLAN": r.PLAN, "claims": r.n, "rules_payor": r.rules_payor,
               "n_plans": int(r.n_plans), "claim_segment": cseg}
        for i, (_, c) in enumerate(cand.iterrows(), 1):
            out |= {f"cand{i}_plan_id": c.PLAN_ID, f"cand{i}_name": c.plan_name,
                    f"cand{i}_seg": c.seg, f"cand{i}_score": round(c.score, 2)}
        gap = cand.score.iloc[0] - (cand.score.iloc[1] if len(cand) > 1 else 0)
        out["confidence"] = "high" if gap >= 0.2 else "check"
        out["approved_plan_id"] = ""          # filled by a person
        out["note"] = ""
        rows.append(out)
    cands = pd.DataFrame(rows).sort_values("claims", ascending=False)
    if len(cands):
        print("\nMulti-plan claim names: confidence high vs check (claims):\n",
              cands.groupby("confidence").claims.agg(["size", "sum"]).to_string())

    with pd.ExcelWriter(a.out) as w:
        size.to_excel(w, sheet_name="sizing")
        multi.to_excel(w, sheet_name="multi_plan_payors")
        cands.to_excel(w, sheet_name="multi_plan_candidates", index=False)
        single.to_excel(w, sheet_name="single_plan_auto", index=False)
        plans[plans.xref_id.isin(multi.index)].sort_values(["xref_id", "PLAN_ID"])[
            ["xref_id", "PLAN_ID", "payer_name", "plan_name", "seg", "codes", "rh_payer_id"]
        ].to_excel(w, sheet_name="rules_plans_multi", index=False)
    print(f"\nWrote {a.out}")


if __name__ == "__main__":
    main()
