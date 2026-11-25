import pandas as pd, numpy as np
t = pd.read_parquet("exp_feat_core_s42/test_preds.parquet"); p = [c for c in t if c.startswith("p_")][0]
fo = next(c for c in ["FILING_ORDER_C", "IS_PRIMARY"] if c in t)
S = t.groupby(["CSN", "FINAL_CPT"]).agg(y=("y", "max"), s=(p, "max")).reset_index()
top = t.sort_values(p, ascending=False).drop_duplicates(["CSN", "FINAL_CPT"])[["CSN", "FINAL_CPT", fo]]
S = S.merge(top, on=["CSN", "FINAL_CPT"]); cut = np.quantile(S.s[S.y == 1], 0.15)
S["cm"] = np.select([(S.y == 1) & (S.s >= cut), (S.y == 0) & (S.s >= cut), S.y == 1], ["TP", "FP", "FN"], "TN")
print(fo, cut); print(pd.crosstab(S[fo], S.cm, margins=True))
