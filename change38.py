import pandas as pd, numpy as np
from sklearn.metrics import average_precision_score
t = pd.read_parquet("exp_d1_full/test_preds.parquet")
t["p"] = t.p_lgb_rules_full
x = t[t.FINAL_CPT.astype(str) == "99214"]
print(f"99214 test lines {len(x):,} | PA rate {x.y.mean():.2%} | mean score {x.p.mean():.3f} | AP within 99214 {average_precision_score(x.y, x.p):.3f}")
g = x.groupby("PAYOR_PLAN_CLEAN").agg(lines=("y","size"), pa_rate=("y","mean"), mean_score=("p","mean"))
g["AP"] = x.groupby("PAYOR_PLAN_CLEAN").apply(lambda d: average_precision_score(d.y, d.p) if 0 < d.y.mean() < 1 else np.nan)
print(g[g.lines >= 300].sort_values("pa_rate", ascending=False).head(15).round(3).to_string())
# share of the model's errors (at 90% recall) that are 99214
v = t.groupby("CSN").agg(y=("y","max"), p=("p","max")).sort_values("p", ascending=False)
cut = v.p.iloc[int(np.argmax((v.y.cumsum() / v.y.sum()).values >= 0.9))]
top = t.sort_values("p", ascending=False).drop_duplicates("CSN").set_index("CSN").FINAL_CPT.astype(str)
fn = v[(v.p < cut) & (v.y == 1)].index; fp = v[(v.p >= cut) & (v.y == 0)].index
print(f"99214 as the top line: {(top[fn]=='99214').mean():.1%} of missed PA visits | {(top[fp]=='99214').mean():.1%} of false alarms")



import pandas as pd, numpy as np, pyarrow.fs as pafs
raw_p = "cli-ml-data-prod/datalake/ai-prior-auth/09282026/v3/ai_prior_auth_09282026_v3_2025.parquet"
r = pd.read_parquet(raw_p, filesystem=pafs.S3FileSystem(),
                    columns=["CSN","LINE","FINAL_CPT","LINE_AUTHORIZATION_NUMBER","CARC","RARC","FILING_ORDER_C"])
r = r[pd.to_numeric(r.FILING_ORDER_C, errors="coerce") == 1]
auth = r.LINE_AUTHORIZATION_NUMBER.astype(str).str.strip()
r["auth"] = auth.where(~auth.str.upper().isin(["", "NAN", "NONE", "NULL"]) & ~auth.str.upper().str.contains(r"\bNAR\b|NO AUTH|NOT REQ"))
code = r.CARC.astype(str).str.extract(r"(\d+)")[0]
r["denial"] = code.isin(["15","39","197","198","210","284","296","302"]) | r.RARC.astype(str).str.contains("M62|N54", na=False)

# one row per claim line: its auth number(s) and whether ANY remit row had a PA denial
ln = r.groupby(["CSN","LINE"]).agg(cpt=("FINAL_CPT", "first"), auth=("auth", "first"), denial=("denial", "max")).reset_index()
shared = ln.dropna(subset=["auth"]).groupby(["CSN","auth"]).LINE.transform("nunique") > 1
ln["auth_shared"] = False; ln.loc[shared.index, "auth_shared"] = shared
ln["evidence"] = np.select([ln.denial, ln.auth.notna() & ~ln.auth_shared, ln.auth.notna() & ln.auth_shared],
                           ["own denial", "own unique auth", "shared auth only"], "none")
true = ln[ln.evidence != "none"]
print("TRUE-evidence lines by source:\n", true.evidence.value_counts().to_string())

per = true.groupby("cpt").evidence.value_counts().unstack(fill_value=0)
per["total"] = per.sum(axis=1)
per["shared_only_share"] = per.get("shared auth only", 0) / per.total
per = per[per.total >= 200].sort_values("shared_only_share", ascending=False)
pd.set_option("display.width", 250)
print("\nCPTs whose 'PA' is mostly inherited (shared auth only):\n", per.head(25).round(3).to_string())
print("\nCPTs with mostly their own evidence:\n", per.tail(15).round(3).to_string())
inherited = per[per.shared_only_share >= 0.9].index
print(f"\nLines that are PA only via a shared auth on CPTs that are >=90% inherited: "
      f"{true[true.cpt.isin(inherited) & (true.evidence=='shared auth only')].shape[0]:,} "
      f"of {len(true):,} TRUE-evidence lines")
