d["same"] = d.PRODUCT_NAME.astype(str).str.upper().str.strip() == d.PLAN.astype(str).str.upper().str.strip()
print("PRODUCT_NAME equals PLAN:", f"{d.same.mean():.1%}")
x = d[~d.same & ~d.PRODUCT_NAME.astype(str).str.contains("Unspecified", na=True)]
print(x.groupby(["PAYOR","PLAN","PRODUCT_NAME"], observed=True).size().sort_values(ascending=False).head(25).to_string())
print(d.groupby("PAYOR", observed=True).PRODUCT_NAME.apply(lambda s: s.astype(str).str.contains("Unspecified").mean())
       .sort_values(ascending=False).head(15).to_string())
