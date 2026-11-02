p = x.groupby("PAT_ID").y.agg(visits="size", pa="sum")
pa_p = p[p.pa > 0]
print("patients with any PA:", len(pa_p), "| PA lines:", int(p.pa.sum()))
print(pa_p.pa.value_counts().sort_index().to_string())            # how many patients have 1, 2, 3... PA visits
print("share of PA lines from patients with 3+ PA visits:", round(pa_p.pa[pa_p.pa >= 3].sum() / p.pa.sum(), 3))
print("PA rate of a patient's later visits, given an earlier PA visit:")
x = x.sort_values(["PAT_ID", "SERVICEDATE"])
x["prior_pa"] = x.groupby("PAT_ID").y.transform(lambda s: s.shift().cummax()).fillna(0)
print(x.groupby("prior_pa").y.agg(lines="size", pa="sum", pa_rate="mean"))
