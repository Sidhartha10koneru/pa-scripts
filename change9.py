print("ID pairs with >1 payor/plan name:", (names_per_id.max(axis=1) > 1).sum(), "of", len(names_per_id))
ids_per_name = d.groupby(["PAYOR","PLAN"])[["BUCKET_PAYOR_ID","BUCKET_PLAN_ID"]].nunique()
print("name pairs with >1 ID pair:", (ids_per_name.max(axis=1) > 1).sum(), "of", len(ids_per_name))
print(d.groupby(["BUCKET_PAYOR_ID","BUCKET_PLAN_ID"]).agg(PAYOR=("PAYOR","first"),PLAN=("PLAN","first"),n=("PLAN","size")).sort_values("n",ascending=False).head(15))
