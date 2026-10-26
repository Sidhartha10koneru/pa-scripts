c = x[x.has_codes].copy()
top = c.CARC.astype(str).value_counts().head(15).index
c["code"] = c.CARC.astype(str).where(c.CARC.astype(str).isin(top), "OTHER")

# 1) code mix by month: do 23 / MA18 / MA07 / MA15 shrink in recent months?
print(pd.crosstab(c.month, c.code, normalize="index").round(3).to_string())

# 2) how each top code is labelled (which list it sits on)
print(pd.crosstab(c.code, c.label, normalize="index").round(3).to_string())
