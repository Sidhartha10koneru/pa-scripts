t["prim"] = t[fo].astype(str).isin(["1", "1.0"])
S0 = t.groupby(["CSN", "FINAL_CPT"]).agg(y=("y", "max"), s_all=(p, "max"), has_prim=("prim", "max")).reset_index()
S0 = S0.join(t[t.prim].groupby(["CSN", "FINAL_CPT"])[p].max().rename("s_prim"), on=["CSN", "FINAL_CPT"])
S0["s_new"] = S0.s_prim.where(S0.has_prim, S0.s_all)
print(S0.has_prim.value_counts())
for col in ["s_all", "s_new"]:
    cut = np.quantile(S0[col][S0.y == 1], 0.15); f = S0[col] >= cut
    tp, fp = int((f & (S0.y == 1)).sum()), int((f & (S0.y == 0)).sum())
    print(col, round(cut, 4), "TP", tp, "FP", fp, "precision", round(tp / (tp + fp), 4))
