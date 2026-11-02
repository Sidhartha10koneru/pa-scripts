x["L1"] = x.LEVEL1_LABEL.astype(str)
x["L2"] = x.LEVEL2_LABEL.astype(str)
x["group"] = np.select([~x.pat_has_pa, x.prior_pa == 1], ["patient never PA", "after a PA visit"], "PA patient, before/at PA")

# 1) how each FALSE was decided, by group
f = x[x.y == 0]
print(pd.crosstab([f.L1, f.L2], f.group, margins=True))

# 2) the same for TRUE (sanity check: should be L1 genuine auth or L2 PA code)
print(pd.crosstab([x[x.y == 1].L1, x[x.y == 1].L2], x[x.y == 1].group, margins=True))

# 3) what the raw auth field and CARC actually contain on FALSE lines
print(f.LEVEL1_RAW_LABEL.astype(str).value_counts().head(10))
print(f.LINE_AUTHORIZATION_NUMBER.astype(str).str.upper().value_counts().head(15))
print(f.CARC.astype(str).value_counts().head(15))
