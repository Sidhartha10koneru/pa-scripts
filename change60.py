x = load_cell("97112", "EMBLEM HEALTH / GHI | NYCE PPO PLAN", "GP").sort_values(["PAT_ID", "SVC"])
p = x.groupby("PAT_ID").agg(first_visit=("SVC", "min"), last_visit=("SVC", "max"), visits=("y", "size"), pa_visits=("y", "sum"))
p["first_pa"] = x[x.y == 1].groupby("PAT_ID").SVC.min()
x = x.join(p.first_pa, on="PAT_ID")

pa_pat = p[p.pa_visits > 0]
before = x[x.first_pa.notna() & (x.SVC < x.first_pa)]
print(f"patients with PA: {len(pa_pat)} | their first PA by month:\n{pa_pat.first_pa.dt.to_period('M').value_counts().sort_index().to_string()}")
print(f"\nFALSE visits before the first PA (same patients): {len(before)}")
print(" of which before April 2026:", int((before.SVC < '2026-04-01').sum()), "| April or later:", int((before.SVC >= '2026-04-01').sum()))
print("visits before first PA, per patient:\n", before.groupby('PAT_ID').size().value_counts().sort_index().to_string())

after = x[x.SVC >= "2026-04-01"]
pp = after.groupby("PAT_ID").y.max()
print(f"\npatients with PT visits from April: {len(pp)} | ever PA: {int(pp.sum())} | never PA: {int((pp == 0).sum())}")
print("visits from April by never-PA patients:", int(after[after.PAT_ID.isin(pp[pp == 0].index)].shape[0]))
