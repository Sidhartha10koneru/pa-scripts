mid = t[(t.p_lgb_rules_full > 0.25) & (t.p_lgb_rules_full < 0.35)].sample(6, random_state=1)
Xm = apply_categories(mid, cats, feats)
cm = pd.DataFrame(m.predict(Xm, pred_contrib=True), columns=feats + ["average"], index=mid.index)
for i in mid.index:
    top = cm.loc[i, feats].sort_values(key=abs, ascending=False).head(3)
    print(mid.loc[i, ["FINAL_CPT","PAYOR_PLAN_CLEAN","RULE_PLAN_SHARE","y"]].to_dict(), "|", top.round(2).to_dict())
