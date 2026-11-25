# 4.6 One patient's course: the 4 history columns build up visit by visit, and both models' decisions
cand = X[(X.core == "FN") & (X.with_history == "TP") & (X.PAH_ANY_90D >= 3)]    # PA the core model missed, history model caught
csn = cand.index[0].split("|")[0]                                                # try cand.index[1], [2], ... for other patients
ids = pd.read_parquet(V11, columns=["CSN", "PAT_ID"], filesystem=fs)
pat = ids.loc[ids.CSN.astype(str).str.replace(r"\.0$", "", regex=True) == csn, "PAT_ID"].iloc[0]; del ids
p = pd.read_parquet(V11, columns=["CSN", "FINAL_CPT", "SERVICEDATE", "LABEL_V6_PAX", "PAH_CPT_90D", "PAH_FAM_90D", "PAH_ANY_90D", "PAH_FAM_365D"],
                    filters=[("PAT_ID", "==", pat)], filesystem=fs)
p = p[~p.FINAL_CPT.astype(str).str.fullmatch(r"\d{4}F")]
p["date"] = pd.to_datetime(p.SERVICEDATE)
g = p.assign(k=key(p)).groupby("k").agg(date=("date", "min"), CPT=("FINAL_CPT", "first"),
        PA=("LABEL_V6_PAX", lambda s: s.astype(str).str.strip().str.upper().eq("TRUE").any()),
        same_CPT_90d=("PAH_CPT_90D", "max"), same_family_90d=("PAH_FAM_90D", "max"),
        any_90d=("PAH_ANY_90D", "max"), same_family_365d=("PAH_FAM_365D", "max"))
g = g.join(X[["score", "core", "score_hist", "with_history"]]).sort_values(["date", "CPT"])
g.insert(1, "family", g.CPT.map(family)); g.insert(0, "day", (g.date - g.date.min()).dt.days)
print(f"cut-offs: core {cut:.3f} | history model {cut_h:.3f}   (scores only for test months May-Jul)")
g.drop(columns="date").reset_index(drop=True).round(3)
