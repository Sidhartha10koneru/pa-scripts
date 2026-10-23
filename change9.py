pd.set_option("display.max_colwidth", 120); pd.set_option("display.width", 250)
d[["PAYOR","PLAN"]] = d[["PAYOR","PLAN"]].astype(str)
print("plan ID missing:", d.BUCKET_PLAN_ID.isna().mean())

s = d.groupby(["BUCKET_PAYOR_ID","BUCKET_PLAN_ID"], dropna=False).agg(
    n=("PLAN","size"), n_payor=("PAYOR","nunique"), n_plan=("PLAN","nunique"),
    payors=("PAYOR", lambda x: sorted(x.unique())[:4]), plans=("PLAN", lambda x: sorted(x.unique())[:4]))
print(f"ID pairs {len(s)} | >1 payor name {(s.n_payor>1).sum()} ({s.loc[s.n_payor>1,'n'].sum()/s.n.sum():.1%} of lines)"
      f" | >1 plan name {(s.n_plan>1).sum()} ({s.loc[s.n_plan>1,'n'].sum()/s.n.sum():.1%} of lines)")
print("\n-- ID pairs with different PAYOR names (bad if different companies):")
print(s[s.n_payor>1].sort_values("n", ascending=False).head(15).to_string())
print("\n-- ID pairs with same payor but several PLAN names (ok if renames):")
print(s[(s.n_payor==1)&(s.n_plan>1)].sort_values("n", ascending=False).head(10).to_string())

nm = d.groupby(["PAYOR","PLAN"]).agg(n=("PLAN","size"), k=("BUCKET_PAYOR_ID","nunique"),
      ids=("BUCKET_PAYOR_ID", lambda x: sorted(x.unique())[:5]))
print(f"\nname pairs {len(nm)} | with >1 payor ID {(nm.k>1).sum()} ({nm.loc[nm.k>1,'n'].sum()/nm.n.sum():.1%} of lines)")
print(nm[nm.k>1].sort_values("n", ascending=False).head(10).to_string())
