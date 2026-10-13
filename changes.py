z = pd.read_parquet(path, columns=["PAYOR","PLAN","PAT_ZIP"], filesystem=s3)
z = z[z.PAYOR.astype(str).str.startswith("ANTHEM")]
z["zip3"] = z.PAT_ZIP.astype(str).str[:3]
z.groupby("PLAN", observed=True).zip3.agg(lambda s: s.value_counts().head(3).to_dict())
