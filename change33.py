import numpy as np, pandas as pd, pyarrow.fs as pafs
import pa_base_adapter as pb, pa_experiments as pe
pd.set_option("display.width", 250)

# training window, same cleaning as the model (takes ~2 min)
cols = ["CSN","GROUP_LABEL","SERVICEDATE","FILING_ORDER_C","FINAL_CPT","FINAL_MODIFIERS","PLACE_OF_SERVICE_CODE","PAYOR","PLAN"]
d = pe.add_dates(pb.load([2025, 2026], cols, 1.0, pafs.S3FileSystem()))
t = d[d.SVC_DATE < "2026-02-01"].reset_index(drop=True); del d
t["FINAL_CPT"] = t.FINAL_CPT.astype(str)

# STEP 0: overall rate and "know nothing" entropy
base = t.y.mean()
H0 = -(base*np.log2(base) + (1-base)*np.log2(1-base))
print(f"STEP 0  lines {len(t):,}  PA {int(t.y.sum()):,}  overall rate {base:.4%}  entropy with no info {H0:.4f} bits\n")

# STEP 1: split visits into two halves (same hash as the script)
t["half"] = np.where(pd.util.hash_pandas_object(t.CSN, index=False) % 2 == 0, "A", "B")

# STEP 2-4 for a few real CPTs: learn rate on A, score B, measure surprise
m = 20
rows = []
for cpt in ["99214", "70553", "97110", "3074F"]:
    a = t[(t.FINAL_CPT == cpt) & (t.half == "A")]; b = t[(t.FINAL_CPT == cpt) & (t.half == "B")]
    rate_A = (a.y.sum() + m*base) / (len(a) + m)                 # smoothed rate learned on half A
    pa_b, no_b = int(b.y.sum()), int(len(b) - b.y.sum())
    s_cpt  = (pa_b * -np.log2(rate_A) + no_b * -np.log2(1 - rate_A)) / len(b)   # surprise using CPT
    s_none = (pa_b * -np.log2(base)   + no_b * -np.log2(1 - base))   / len(b)   # surprise knowing nothing
    rows.append(dict(CPT=cpt, A_lines=len(a), A_PA=int(a.y.sum()), rate_learned_on_A=rate_A,
                     B_lines=len(b), B_PA=pa_b, surprise_if_PA=-np.log2(rate_A), surprise_if_not=-np.log2(1-rate_A),
                     avg_surprise_with_CPT=s_cpt, avg_surprise_no_info=s_none))
print("STEPS 2-4 for example CPTs (learn on half A, score half B)\n", pd.DataFrame(rows).round(4).to_string(index=False), "\n")

# STEP 5: do it for ALL CPTs, both directions, and average over every line
p = np.empty(len(t))
for learn, score in (("A", "B"), ("B", "A")):
    g = t[t.half == learn].groupby("FINAL_CPT").y.agg(["sum", "count"])
    rate = (g["sum"] + m*base) / (g["count"] + m)
    idx = (t.half == score).values
    p[idx] = t.loc[idx, "FINAL_CPT"].map(rate).fillna(base).values
p = np.clip(p, 1e-6, 1 - 1e-6)
left = -(t.y*np.log2(p) + (1-t.y)*np.log2(1-p)).mean()
print(f"STEP 5  average surprise over all {len(t):,} lines using CPT: {left:.4f} bits (vs {H0:.4f} knowing nothing)")
print(f"STEP 6  info gain = 1 - {left:.4f}/{H0:.4f} = {1 - left/H0:.1%}")
