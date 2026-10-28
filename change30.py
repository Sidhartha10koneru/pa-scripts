import numpy as np, pandas as pd
def caught_at(p, n_flags):
    e = pd.DataFrame({"y": t.y.values, "p": p, "c": t.CSN.values}).groupby("c").max().sort_values("p", ascending=False)
    return e.y.iloc[:n_flags].sum() / e.y.sum()
w = pd.read_csv("exp_d2_cells/weights_cpt_plan_k20.csv", index_col="cell")["weight"]
cell = t[["FINAL_CPT","PAYOR_PLAN_CLEAN"]].astype(str).agg("|".join, axis=1)
p_cw = sig(raw + cell.map(w).fillna(0).values)
for n in [5000, 9233, 15000, 26000]:
    print(n, "visits flagged | recall frozen:", round(caught_at(sig(raw), n), 3), "| cell weights:", round(caught_at(p_cw, n), 3))
