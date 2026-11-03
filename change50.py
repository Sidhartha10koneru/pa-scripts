import pyarrow.fs as pafs, fn_root_cause as rc
fs = pafs.S3FileSystem()
u = u.assign(PAT_ID=u.PAT_ID.astype(str), SVC=pd.to_datetime(u.SVC, errors="coerce"))
pats = u.PAT_ID.dropna().unique()
P = pd.concat([rc.read(d, rc.isin(d, "PAT_ID", pats)) for d in (rc.dset(y, fs) for y in [2025, 2026])], ignore_index=True)

# earlier lines of the same patients that were TRUE from a code (TRUE, no auth number)
den = P[(P.L == "TRUE") & P.AUTH.eq("")][["PAT_ID", "SVC", "FINAL_CPT"]].drop_duplicates()
m = u[["CSN", "PAT_ID", "SVC", "FINAL_CPT"]].merge(den, on="PAT_ID", suffixes=("", "_d"))
m = m[m.SVC_d < m.SVC - pd.Timedelta(days=45)]                  # known before the visit (45-day claim lag)
m["same_cpt"] = m.FINAL_CPT == m.FINAL_CPT_d
pr = m.groupby("CSN").agg(n_prior=("SVC_d", "size"), same_cpt=("same_cpt", "any"))
u["prior_denial"] = u.CSN.isin(pr.index)
u["prior_denial_same_cpt"] = u.CSN.map(pr.same_cpt).fillna(False).astype(bool)

print(f"FN {len(u)} | patients {u.PAT_ID.nunique()}")
print(pd.crosstab(u.t1_source, [u.prior_denial, u.prior_denial_same_cpt], margins=True))
print("\nFN per patient:\n", u.groupby("PAT_ID").size().value_counts().sort_index().head(10))
print("\nby plan (top 10):\n", pd.crosstab(u.PAYOR_PLAN_CLEAN, u.prior_denial).assign(n=lambda t: t.sum(1))
      .sort_values("n", ascending=False).head(10))
