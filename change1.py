import numpy as np, pandas as pd
def svc(p):
    t = pd.read_parquet(p); c = [x for x in t.columns if x.startswith("p_")][0]
    s = t.groupby(["CSN", "FINAL_CPT"]).agg(y=("y", "max"), s=(c, "max")).reset_index()
    return s[~s.FINAL_CPT.astype(str).str.fullmatch(r"\d{4}F")]
for a, b in [("exp_nofilter_2022", "exp_qc_exclude_rv"), ("exp_keep_s7", "exp_exclude_s7")]:
    A, B = svc(f"{a}/test_preds.parquet"), svc(f"{b}/test_preds.parquet")
    cut = np.quantile(A.s[A.y == 1], 0.20); budget = int(((A.s >= cut) & (A.y == 0)).sum())   # base FP at 80% recall
    neg = np.sort(B.s[B.y == 0].to_numpy())[::-1]; cut_b = neg[budget - 1]                    # B's cut-off with the same FP
    print(f"{a} vs {b}: FP budget {budget:,} | TP base {int(((A.s >= cut) & (A.y == 1)).sum()):,} "
          f"| TP exclude {int(((B.s >= cut_b) & (B.y == 1)).sum()):,}")
