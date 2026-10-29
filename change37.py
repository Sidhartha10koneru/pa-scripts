raw_p = "cli-ml-data-prod/datalake/ai-prior-auth/09282026/v3/ai_prior_auth_09282026_v3_2025.parquet"
csn75 = t.loc[t.source == "neither (check!)", "CSN"].unique()
r = pd.read_parquet(raw_p, filesystem=pafs.S3FileSystem(),
                    columns=["CSN","LINE","FINAL_CPT","LINE_AUTHORIZATION_NUMBER","CARC","RARC","FILING_ORDER_C"])
r = r[r.CSN.isin(csn75) & (r.FINAL_CPT.astype(str) == "99214")]
pd.set_option("display.width", 250)
print(len(csn75), "visits;", len(r), "raw remit rows")
print(r.sort_values(["CSN","LINE"]).head(40).to_string())
print(r.CARC.astype(str).value_counts().head(15).to_string())
