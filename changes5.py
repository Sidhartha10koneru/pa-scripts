for p in ["AARP MEDICARE", "BCBS OF NM"]:
    g = c[(c.PAYOR == p) & ~c.companion]
    t = g.groupby("PLAN").apply(lambda x: pd.Series({
        "lines": len(x), "pa": int(x.is_true.sum()),
        "pa_cov": x.loc[x.is_true, "on_own"].mean(),
        "nonpa_cov": x.loc[~x.is_true, "on_own"].mean()}))
    print("\n", p); print(t.sort_values("lines", ascending=False).head(15).round(3).to_string())
