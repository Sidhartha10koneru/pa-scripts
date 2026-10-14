f = pd.read_parquet(path, columns=["PAYOR", "FILING_ORDER_C", "GROUP_LABEL"], filesystem=s3)
print(f.FILING_ORDER_C.value_counts(dropna=False).head(10))
print(pd.crosstab(f.FILING_ORDER_C, f.GROUP_LABEL.astype(str), normalize="index").round(3))
print(pd.crosstab(f.FILING_ORDER_C, f.PAYOR.astype(str).str.upper().eq("MEDICARE")))
