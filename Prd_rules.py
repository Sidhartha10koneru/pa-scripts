# save as profile_rules.py, run:  python profile_rules.py ~/Downloads/Prod_Globals
import sys, glob, os, re, pandas as pd
d = sys.argv[1]
pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 40)

def show(f, n=5):
    t = pd.read_csv(f, dtype=str, low_memory=False)
    print(f"\n=== {os.path.basename(f)}  rows={len(t):,}  cols={len(t.columns)}")
    print(list(t.columns))
    print(t.head(n).to_string(index=False))
    for c in t.columns:
        print(f"  {c}: {t[c].nunique()} distinct, {t[c].isna().mean():.0%} empty")
    return t

for name in ["PayerIdPlanId_Map.csv", "payerProfile_Map.csv", "AllCPTCodes.csv"]:
    p = os.path.join(d, name)
    if os.path.exists(p):
        show(p)

files = sorted(glob.glob(os.path.join(d, "payerProfile_ScreeningRules_*.csv")))
print(f"\n\n{len(files)} ScreeningRules files")
summ = []
for f in files:
    t = pd.read_csv(f, dtype=str, low_memory=False)
    summ.append({"profile": re.findall(r"_(\d+)\.csv$", f)[0], "rows": len(t), "cols": "|".join(t.columns)})
s = pd.DataFrame(summ)
print("column layouts:", s.cols.nunique())
print(s.cols.value_counts().head(5).to_string())
print(s.rows.describe().round(0).to_string())
print("\nlargest:", s.sort_values("rows", ascending=False).head(5).to_string(index=False))
show(files[0], 10)          # one full example
