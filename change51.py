s1 = s[s.L.isin(["TRUE","FALSE"])].copy()                       # labelled lines from your 2026 sample
n = s1.groupby("CSN").FINAL_CPT.transform("nunique")
s1["visit"] = np.where(n == 1, "alone on claim", "with other services")
r = s1.assign(y=s1.L.eq("TRUE")).groupby(["FINAL_CPT", "visit"]).y.agg(["mean", "size"]).unstack()
r = r[(r["size"] >= 50).all(1)]
r["carried_x"] = r[("mean", "with other services")] / r[("mean", "alone on claim")].replace(0, np.nan)
print(r.sort_values(("size", "with other services"), ascending=False).head(25).round(3))
