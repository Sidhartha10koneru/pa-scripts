g = t.assign(listed=t.RULE_PLAN_SHARE.eq(1), notlisted=t.RULE_PLAN_SHARE.eq(0))
s = g.groupby("FINAL_CPT").apply(lambda d: pd.Series({
        "n_listed": d.listed.sum(), "pa_listed": d.loc[d.listed, "y"].mean(),
        "n_not": d.notlisted.sum(), "pa_not": d.loc[d.notlisted, "y"].mean()}))
s = s[(s.n_listed >= 200) & (s.n_not >= 200)]
s["lift"] = s.pa_listed / s.pa_not.replace(0, np.nan)
print(len(s), "CPTs | listing matters (lift>2):", (s.lift > 2).sum(), "| listing ~no effect (0.5–2):", s.lift.between(0.5, 2).sum())
print(s.sort_values("n_listed", ascending=False).head(20).round(3).to_string())
