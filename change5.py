E = X[X.family == "EM"]
pd.crosstab(E.core, E.with_history, margins=True)                    # how E/M outcomes moved
E.assign(any_hist=E.PAH_ANY_90D > 0).groupby("any_hist").y.agg(services="size", pa_rate="mean")   # PA rate with vs without any recent PA
fixed = E[(E.core == "FP") & (E.with_history == "TN") & (E.PAH_ANY_90D == 0)]
cpt, payer = fixed.groupby(["FINAL_CPT", "payer"]).size().idxmax()
ex = E[(E.FINAL_CPT == cpt) & (E.payer == payer)]
pd.concat([ex[(ex.core == "FP") & (ex.with_history == "TN")].head(4),
           ex[(ex.y == 1) & (ex.with_history == "TP") & (ex.PAH_ANY_90D > 0)].head(4)])[
    ["FINAL_CPT", "payer", "PAH_ANY_90D", "y", "score", "core", "score_hist", "with_history"]].round(3)
