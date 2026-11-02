show = ["SPECIALTY", "DEPARTMENT", "PCPNAME", "SERVICEPROVIDER_NAME", "Primary_DX", "SERVICEDATE", "PAT_ID", "FILING_ORDER_C"]
show = [c for c in show if c in x.columns]
print(x[x.y == 1][show].head(15).to_string())
print(x[x.y == 0][show].sample(15, random_state=0).to_string())

c = "SPECIALTY"
cmp = pd.crosstab(x[c].astype(str), x.y, normalize="columns").rename(columns={0: "share_of_nonPA", 1: "share_of_PA"})
cmp["diff"] = cmp.share_of_PA - cmp.share_of_nonPA
print(cmp.sort_values("diff", ascending=False).head(10).round(3))

both = x.groupby("PAT_ID").y.agg(["min", "max"])
pats = both[(both["min"] == 0) & (both["max"] == 1)].index
print(len(pats), "patients with both PA and non-PA 99214 visits")
p = x[x.PAT_ID.isin(pats)].sort_values(["PAT_ID", "SERVICEDATE"])
print(p[["PAT_ID", "SERVICEDATE", "y"] + [c for c in show if c not in ("PAT_ID", "SERVICEDATE")]].head(40).to_string())
