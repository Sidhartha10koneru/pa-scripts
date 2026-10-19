"""
analyse_prod_rules.py - understand the Prod_Globals rules files before any mapping.

Run:
    python analyse_prod_rules.py Prod_Globals
    python analyse_prod_rules.py Prod_Globals --old REQ-260378.xlsx     # also compare with the old rules Excel

Prints a report (save it: ... > rules_analysis.txt) and writes ./rules_analysis/:
    all_rules.parquet          every ScreeningRules row in one table (+ file_profile, code_type)
    plan_summary.csv           one row per rule profile (plan): payer, ids, size, code mix, vendor, dates
    exception_texts.csv        every distinct EXCEPTION_CODES text: rows, plans, auto category
    payer_profile_values.csv   value counts of the categorical columns of payerProfile_Map
    old_vs_new.csv             (with --old) old PAYOR_CODE vs new ids, and code-set overlap
"""
import argparse
import glob
import os
import re

import pandas as pd

pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_colwidth", 70)


def h(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


def code_type(c):
    c = str(c).strip().upper()
    if re.fullmatch(r"\d{5}", c):
        return "CPT (numeric)"
    if re.fullmatch(r"\d{4}F", c):
        return "CPT Cat II (####F)"
    if re.fullmatch(r"\d{4}T", c):
        return "CPT Cat III (####T)"
    if re.fullmatch(r"\d{4}U", c):
        return "PLA lab (####U)"
    if re.fullmatch(r"\d{4}[A-Z]", c):
        return "CPT other suffix"
    if re.fullmatch(r"[A-V]\d{4}", c):
        return "HCPCS II (letter+4)"
    return "other"


def exception_category(t):
    s = str(t).lower()
    if not s or s == "nan":
        return "empty"
    if re.search(r"not required|no (prior )?auth|does not require|no pa\b|not need", s):
        return "NOT required"
    cond = re.search(r"\bonly\b|\bwhen\b|\bif\b|\bexcept|unless|diagnos|\bdx\b|place of service|\bpos\b|"
                     r"\bage\b|years|inpatient|outpatient|units|per |\$|over|greater|more than", s)
    if re.search(r"required|requires|require", s):
        return "required - conditional" if cond else "required"
    if re.search(r"evicore|carelon|aim|optum|cohere|turningpoint|vendor|refer", s):
        return "routed to vendor"
    return "other / unclear"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--old", default=None, help="old rules Excel (REQ-260378.xlsx), sheet 'Plans with Services'")
    ap.add_argument("--out", default="rules_analysis")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rd = lambda n: pd.read_csv(os.path.join(a.folder, n), dtype=str, low_memory=False)

    # ------------------------------------------------------------------ 1. the rule files
    h("1. ScreeningRules files: structure")
    files = sorted(glob.glob(os.path.join(a.folder, "payerProfile_ScreeningRules_*.csv")))
    parts = []
    for f in files:
        t = pd.read_csv(f, dtype=str, low_memory=False)
        t["file_profile"] = re.findall(r"_(\d+)\.csv$", f)[0]
        parts.append(t)
    r = pd.concat(parts, ignore_index=True)
    for c in r.columns:
        r[c] = r[c].str.strip()
    r["code_type"] = r["MEDICAL_CODES"].map(code_type)
    r.to_parquet(f"{a.out}/all_rules.parquet", index=False)
    print(f"files: {len(files)} | total rows: {len(r):,} | distinct codes overall: {r.MEDICAL_CODES.nunique():,}")
    print("PLAN_ID equals the number in the file name on every row:", bool((r.PLAN_ID == r.file_profile).all()))
    per_file = r.groupby("file_profile").agg(**{k: (k, "nunique") for k in
               ["PAYER_NAME", "PLAN_NAME", "RH_PAYER_ID", "XREF_ID", "PAYER_XREF_ID", "PAYER_ALIAS", "GLOBAL_PAYER_DATABASE_ID"]})
    print("files where an identity column has more than one value:")
    print((per_file > 1).sum().to_string())
    print("\nis XREF_ID == PAYER_XREF_ID == PAYER_ALIAS on every row?",
          bool((r.XREF_ID == r.PAYER_XREF_ID).all()), bool((r.XREF_ID == r.PAYER_ALIAS).all()))

    # ------------------------------------------------------------------ 2. one row per plan
    h("2. Plans (rule profiles)")
    first = lambda s: s.mode().iat[0] if s.notna().any() else None
    plan = r.groupby("file_profile").agg(
        payer_name=("PAYER_NAME", first), plan_name=("PLAN_NAME", first),
        rh_payer_id=("RH_PAYER_ID", first), xref_id=("XREF_ID", first), payer_alias=("PAYER_ALIAS", first),
        global_payer_db_id=("GLOBAL_PAYER_DATABASE_ID", first),
        rows=("MEDICAL_CODES", "size"), codes=("MEDICAL_CODES", "nunique"),
        exception_texts=("EXCEPTION_CODES", "nunique"),
        um_vendor=("UM_VENDOR", first), external_route=("EXTERNAL_ROUTE", first),
        revenue_benefit_manager=("REVENUE_BENEFIT_MANAGER", first),
        alias_status=("PAYER_ALIAS_STATUS", first), last_updated_max=("LAST_UPDATED", "max"))
    plan["dup_rows"] = plan.rows - plan.codes
    mix = pd.crosstab(r.file_profile, r.code_type, normalize="index").round(3)
    plan = plan.join(mix)
    plan.to_csv(f"{a.out}/plan_summary.csv")
    print(f"plans: {len(plan)} | payer names: {plan.payer_name.nunique()} | RH payer ids: {plan.rh_payer_id.nunique()} "
          f"| xref ids: {plan.xref_id.nunique()}")
    print("\nplans per payer name (top 15):")
    print(plan.payer_name.value_counts().head(15).to_string())
    print("\ncodes per plan:"); print(plan.codes.describe().round(0).to_string())
    print("\nplans with repeated codes (same code more than once):", int((plan.dup_rows > 0).sum()))
    print("\nplans with a UM vendor / external route / revenue benefit manager:",
          int(plan.um_vendor.notna().sum()), int(plan.external_route.notna().sum()), int(plan.revenue_benefit_manager.notna().sum()))
    print(plan.um_vendor.value_counts().head(10).to_string())
    print("\ncode type mix, all rows:"); print(r.code_type.value_counts(normalize=True).round(3).to_string())
    print("\nlargest plans:"); print(plan.sort_values("rows", ascending=False).head(8)[["payer_name", "plan_name", "rows", "codes"]].to_string())

    # ------------------------------------------------------------------ 3. what the rows say
    h("3. EXCEPTION_CODES text (the condition attached to each code)")
    ex = r.groupby("EXCEPTION_CODES", dropna=False).agg(rows=("MEDICAL_CODES", "size"), plans=("file_profile", "nunique"))
    ex["category"] = [exception_category(t) for t in ex.index]
    ex = ex.sort_values("rows", ascending=False)
    ex.to_csv(f"{a.out}/exception_texts.csv")
    print(f"distinct texts: {len(ex):,}")
    print("\nrows by auto category (keyword rules - check the CSV before trusting):")
    print(ex.groupby("category").rows.sum().sort_values(ascending=False).to_string())
    print("\ntop 30 texts:"); print(ex.head(30).to_string())

    # ------------------------------------------------------------------ 4. map files
    h("4. PayerIdPlanId_Map")
    m = rd("PayerIdPlanId_Map.csv").apply(lambda s: s.str.strip())
    print(f"rows {len(m)} | payer ids {m.RH_PAYER_ID.nunique()} | plan ids {m.PLAN_ID.nunique()}")
    print("plans per payer id:"); print(m.groupby("RH_PAYER_ID").PLAN_ID.nunique().value_counts().sort_index().to_string())
    print("payer ids per plan id:"); print(m.groupby("PLAN_ID").RH_PAYER_ID.nunique().value_counts().sort_index().to_string())
    fp, mp = set(plan.index), set(m.PLAN_ID)
    print(f"plan ids with a rules file: {len(mp & fp)} | in map, no file: {len(mp - fp)} | file, not in map: {len(fp - mp)}")
    j = m.merge(plan[["rh_payer_id"]].reset_index().rename(columns={"file_profile": "PLAN_ID"}), on="PLAN_ID", how="inner")
    print("map RH_PAYER_ID agrees with the rules file's RH_PAYER_ID:", f"{(j.RH_PAYER_ID == j.rh_payer_id).mean():.1%}")

    h("5. payerProfile_Map")
    p = rd("payerProfile_Map.csv").apply(lambda s: s.str.strip())
    print(f"rows {len(p)} | payers (RH_PAYOR_ID) {p.RH_PAYOR_ID.nunique()} | rows per payer:")
    print(p.groupby("RH_PAYOR_ID").size().value_counts().sort_index().head(10).to_string())
    vals = []
    for c in ["PAYOR_TYPE", "PAYOR_LOB", "PAYOR_STATUS", "PAYOR_XREF_TRX", "PAYOR_XREF_STATUS", "INTERNAL_ROUTE",
              "EXTERNAL_ROUTE", "ID_TYPE", "TRANSACTION_TYPE"]:
        vc = p[c].value_counts(dropna=False)
        print(f"\n{c}:"); print(vc.head(15).to_string())
        vals.append(vc.rename("rows").reset_index().rename(columns={c: "value"}).assign(column=c))
    pd.concat(vals).to_csv(f"{a.out}/payer_profile_values.csv", index=False)
    print("\ncross-tab ID_TYPE x TRANSACTION_TYPE:"); print(pd.crosstab(p.ID_TYPE, p.TRANSACTION_TYPE).to_string())
    pp, mpay, fpay = set(p.RH_PAYOR_ID), set(m.RH_PAYER_ID), set(plan.rh_payer_id.dropna())
    print(f"\npayer ids: in profile map {len(pp)} | in plan map {len(mpay)} | in rule files {len(fpay)} | "
          f"rule-file payers found in profile map {len(fpay & pp)}")
    # which xref in the profile map matches the rule files' XREF_ID?
    xr = set(plan.xref_id.dropna())
    for c in ["PAYOR_XREF_ID", "PAYOR_ALIAS", "PAYOR_XREF_SUB_ID"]:
        print(f"rule-file XREF_ID values found in payerProfile_Map.{c}: {len(xr & set(p[c].dropna()))} of {len(xr)}")
    lob = p.drop_duplicates("RH_PAYOR_ID").set_index("RH_PAYOR_ID")[["PAYOR_LOB", "PAYOR_TYPE"]]
    plan2 = plan.join(lob, on="rh_payer_id")
    print("\nplans by payer LOB / TYPE:"); print(pd.crosstab(plan2.PAYOR_LOB.fillna("-"), plan2.PAYOR_TYPE.fillna("-")).to_string())
    plan2.to_csv(f"{a.out}/plan_summary.csv")

    h("6. AllCPTCodes")
    allc = set(rd("AllCPTCodes.csv").MEDICAL_CODES.str.strip())
    used = set(r.MEDICAL_CODES)
    print(f"reference codes {len(allc):,} | used in rules {len(used):,} | rule codes not in reference {len(used - allc):,}")
    plan2["share_of_all_codes"] = (plan2.codes / len(allc)).round(3)
    print("plans listing > 50% of all codes (probably 'everything needs PA' or a different rule style):",
          int((plan2.share_of_all_codes > 0.5).sum()))
    plan2.to_csv(f"{a.out}/plan_summary.csv")

    # ------------------------------------------------------------------ 7. old Excel
    if a.old:
        h("7. Compare with the old rules Excel")
        o = pd.read_excel(a.old, sheet_name="Plans with Services", dtype=str)
        o["PAYOR_CODE"] = o["PAYOR_CODE"].str.strip()
        old_codes = set(o.PAYOR_CODE.dropna())
        for c in ["xref_id", "payer_alias", "rh_payer_id"]:
            print(f"old PAYOR_CODE found in new {c}: {len(old_codes & set(plan2[c].dropna()))} of {len(old_codes)}")
        coded = o[o.CODING_STATUS.str.strip() == "CODED_LIST"].copy()
        coded["cpt"] = coded.PLAN_CPT_CODES.astype(str).str.replace("_x000D_", ",", regex=False).str.upper().str.split(r"[^0-9A-Z]+")
        coded = coded.explode("cpt"); coded = coded[coded.cpt.str.fullmatch(r"[0-9A-Z]{5}", na=False)]
        old_sets = coded.groupby("PAYOR_CODE").cpt.agg(set)
        new_sets = r.groupby("XREF_ID").MEDICAL_CODES.agg(set)
        rows = []
        for k in sorted(set(old_sets.index) & set(new_sets.index)):
            A, B = old_sets[k], new_sets[k]
            rows.append({"payor_code": k, "old_codes": len(A), "new_codes": len(B), "both": len(A & B),
                         "only_old": len(A - B), "only_new": len(B - A), "jaccard": round(len(A & B) / len(A | B), 3)})
        cmp_ = pd.DataFrame(rows).sort_values("old_codes", ascending=False)
        cmp_.to_csv(f"{a.out}/old_vs_new.csv", index=False)
        print(f"\ncoded payors in old Excel: {len(old_sets)} | also in new rules: {len(cmp_)}")
        if len(cmp_):
            print(cmp_.jaccard.describe().round(3).to_string())
            print(cmp_.head(20).to_string(index=False))
        print("\nold coded payors missing from new:", sorted(set(old_sets.index) - set(new_sets.index))[:40])

    print(f"\nOutputs in ./{a.out}/")


if __name__ == "__main__":
    main()
