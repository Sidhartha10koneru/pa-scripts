S0 = S0.join(t.groupby(["CSN", "FINAL_CPT"]).SVC_DATE.min().dt.to_period("M").rename("m"), on=["CSN", "FINAL_CPT"])
for col in ["s_all", "s_new"]:
    S0["f_" + col] = S0[col] >= np.quantile(S0[col][S0.y == 1], 0.15)
print(S0.groupby("m").apply(lambda g: pd.Series({f"{k}_{n}": int((g["f_" + k] & (g.y == v)).sum())
      for k in ["s_all", "s_new"] for n, v in [("TP", 1), ("FP", 0)]})))
