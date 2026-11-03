x = tr[(tr.FINAL_CPT == "97112") & (tr.FINAL_MODIFIERS == "GP") & (tr.PLACE_OF_SERVICE_CODE == "11")
       & (tr.PAYOR_PLAN_CLEAN == "EMBLEM HEALTH / GHI | NYCE PPO PLAN")]
print(x.groupby(x.SVC_DATE.dt.to_period("M")).y.agg(lines="size", pa="sum", rate="mean"))
te = t[(t.FINAL_CPT == "97112") & (t.FINAL_MODIFIERS == "GP") & (t.PLACE_OF_SERVICE_CODE == "11")
       & (t.PAYOR_PLAN_CLEAN == "EMBLEM HEALTH / GHI | NYCE PPO PLAN")].copy()
te["m"] = pd.to_datetime(te.SERVICEDATE).dt.to_period("M")
print(te.groupby("m").y.agg(lines="size", pa="sum", rate="mean"))
# same CPT at this plan, all codes: is it PT-wide (97110, 97140...) or only 97112?
pt = t[t.FINAL_CPT.str.startswith("971") & (t.PAYOR_PLAN_CLEAN == "EMBLEM HEALTH / GHI | NYCE PPO PLAN")]
print(pt.groupby("FINAL_CPT").y.agg(lines="size", pa="sum", rate="mean").sort_values("lines", ascending=False).head(8))
