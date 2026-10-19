import pandas as pd
r = pd.read_parquet("rules_analysis/all_rules.parquet")
o = pd.read_excel("PATH/REQ-260378.xlsx", sheet_name="Plans with Services", dtype=str)   # set the path

# 1) extra codes: compare the old list with each new plan under the same XREF
c = o[o.CODING_STATUS.str.strip() == "CODED_LIST"].copy()
c["cpt"] = c.PLAN_CPT_CODES.astype(str).str.replace("_x000D_", ",", regex=False).str.upper().str.split(r"[^0-9A-Z]+")
c = c.explode("cpt"); c = c[c.cpt.str.fullmatch(r"[0-9A-Z]{5}", na=False)]
old = c.groupby("PAYOR_CODE").cpt.agg(set)
for x in ["UHC", "AZBCBS", "TRICRE", "OXFORD"]:
    print("\n", x, "old:", len(old[x]))
    for (pid, pn), g in r[r.XREF_ID == x].groupby(["PLAN_ID", "PLAN_NAME"]):
        s = set(g.MEDICAL_CODES)
        print(f"   {pid} {pn[:55]:<55} codes={len(s):>6}  has all old={old[x] <= s}  jaccard={len(s & old[x]) / len(s | old[x]):.2f}")
    extra = r[(r.XREF_ID == x) & ~r.MEDICAL_CODES.isin(old[x])]
    print("   condition text on extra codes:"); print(extra.EXCEPTION_CODES.fillna("(empty)").value_counts().head(5).to_string())

# 2) repeated codes: what differs between the copies?
d = r[r.duplicated(["PLAN_ID", "MEDICAL_CODES"], keep=False)]
print("\nrepeated rows:", len(d))
for col in ["EXCEPTION_CODES", "PAYER_XREF_ID", "UM_VENDOR", "EXTERNAL_ROUTE", "GLOBAL_PAYER_DATABASE_ID"]:
    print(col, "differs within repeats:", f"{(d.groupby(['PLAN_ID','MEDICAL_CODES'])[col].nunique(dropna=False) > 1).mean():.1%}")
print(d[d.PLAN_ID == "1100"].sort_values("MEDICAL_CODES").head(12)[["MEDICAL_CODES", "EXCEPTION_CODES", "PAYER_XREF_ID", "UM_VENDOR"]].to_string())

# 3) PAYER_XREF_ID inside a plan
print("\n", r[r.PLAN_ID == "1100"].PAYER_XREF_ID.value_counts().head(10))

# 4) LOB letters: example payer names
p = pd.read_csv("Prod_Globals/payerProfile_Map.csv", dtype=str)
for l in ["W", "H", "B", "M", "T", "V"]:
    print(l, p[p.PAYOR_LOB == l].PAYOR_NAME.drop_duplicates().head(6).tolist())
