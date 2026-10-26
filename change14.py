x = pd.read_parquet(path, columns=["CSN","SERVICEDATE","GROUP_LABEL","CARC","RARC","LINE_AUTHORIZATION_NUMBER"],
                    filesystem=pafs.S3FileSystem())
x["month"] = pd.to_datetime(x.SERVICEDATE, errors="coerce").dt.to_period("M").astype(str)
x["label"] = x.GROUP_LABEL.astype(str).str.upper().str.strip()
x["has_codes"] = x.CARC.notna() & (x.CARC.astype(str).str.strip() != "")

# 1) FALSE lines: share with no CARC at all (likely "no response yet"), by month
f = x[x.label == "FALSE"]
print(f.groupby("month").has_codes.apply(lambda s: 1 - s.mean()).round(3).rename("FALSE_with_no_codes").to_string())

# 2) the CARC values behind AMBIGUOUS lines (top 20)
print(x.loc[x.label == "AMBIGUOUS", "CARC"].astype(str).value_counts().head(20).to_string())
