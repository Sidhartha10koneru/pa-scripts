# 4.4 (light) One example: core vs history model, same CPT + payer
import gc
H = services(HIST); cut_h = H.score[H.y == 1].quantile(0.15)
h = pd.read_parquet(V11, columns=["CSN", "FINAL_CPT", "SERVICEDATE", "PAH_CPT_90D", "PAH_FAM_90D"], filesystem=fs)
d = pd.to_datetime(h.SERVICEDATE, errors="coerce")
h = h[(d >= "2026-05-01") & (d < "2026-08-01")].drop(columns="SERVICEDATE")
key = lambda D: D.CSN.astype(str).str.replace(r"\.0$", "", regex=True) + "|" + D.FINAL_CPT.astype(str)
h = h.assign(k=key(h)).groupby("k")[["PAH_CPT_90D", "PAH_FAM_90D"]].max()
X = S.assign(k=key(S)).set_index("k")[["FINAL_CPT", "payer", "family", "y", "score"]]
X["score_hist"] = H.assign(k=key(H)).set_index("k").score
X = X.join(h).fillna({"PAH_CPT_90D": 0, "PAH_FAM_90D": 0})
del h, d; gc.collect()
X["core"] = np.where(X.score >= cut, np.where(X.y == 1, "TP", "FP"), np.where(X.y == 1, "FN", "TN"))
X["with_history"] = np.where(X.score_hist >= cut_h, np.where(X.y == 1, "TP", "FP"), np.where(X.y == 1, "FN", "TN"))
fixed = X[(X.core == "FP") & (X.with_history == "TN") & (X.PAH_FAM_90D == 0)]
cpt, payer = fixed.groupby(["FINAL_CPT", "payer"]).size().idxmax()
ex = X[(X.FINAL_CPT == cpt) & (X.payer == payer)]
print(cpt, payer, "| services:", len(ex)); pd.crosstab(ex.core, ex.with_history, margins=True)
