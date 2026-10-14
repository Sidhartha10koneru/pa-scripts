cond_rows = coded.loc[f.any_condition.values, ["PAYOR_CODE","PLAN_CPT_CODES"]]
# explode the conditional rows' codes the same way as exploded_rules_v2
cc = (cond_rows.assign(cpt_code=cond_rows.PLAN_CPT_CODES.astype(str)
        .str.replace("_x000D_", ",").str.split(r"[,\s\n]+"))
        .explode("cpt_code"))
cc["cpt_code"] = cc.cpt_code.str.strip().str.upper()
cond_pairs = set(zip(cc.PAYOR_CODE, cc.cpt_code))

c = pd.read_parquet(path, columns=["PAYOR","PLAN","FINAL_CPT","GROUP_LABEL"], filesystem=s3).astype(str)
c = c.merge(pairs[["PAYOR","PLAN","rules_payor"]].drop_duplicates(["PAYOR","PLAN"]), on=["PAYOR","PLAN"])
c["cpt"] = c.FINAL_CPT.str.strip().str.upper()
own = set(zip(exploded_rules_v2.PAYOR_CODE.astype(str), exploded_rules_v2.cpt_code.astype(str).str.upper()))
c["on_list"] = [(p, x) in own for p, x in zip(c.rules_payor, c.cpt)]
c["conditional"] = [(p, x) in cond_pairs for p, x in zip(c.rules_payor, c.cpt)]
on = c[c.on_list]
print("Listed claim lines:", len(on), "| of which under a conditional rule:", round(on.conditional.mean()*100, 1), "%")
print(on.groupby("rules_payor").conditional.mean().sort_values(ascending=False).head(15).round(2))
