cl = coded.copy()
cl["cpt_code"] = (cl.PLAN_CPT_CODES.astype(str).str.replace("_x000D_", ",", regex=False)
                  .str.upper().str.split(r"[^0-9A-Z]+"))
cl = cl.explode("cpt_code"); cl = cl[cl.cpt_code.str.fullmatch(r"[0-9A-Z]{5}", na=False)]
used = set(pairs.loc[pairs.rules_payor.isin(set(rules_df.PAYOR_CODE.astype(str))), "rules_payor"])
cl = cl[cl.PAYOR_CODE.astype(str).isin(used)]

rows = []
for p, g in cl.groupby("PAYOR_CODE"):
    sets = g.groupby("PLAN_NAME").cpt_code.agg(set)
    if len(sets) < 2: continue
    union = set().union(*sets); inter = set.intersection(*sets)
    vol = int(pairs.loc[pairs.rules_payor == p, "n"].sum())
    rows.append((p, vol, len(sets), " | ".join(f"{k} ({len(v)})" for k, v in sets.items()),
                 round(len(inter) / max(len(union), 1), 2)))
print(pd.DataFrame(rows, columns=["payor","claims","plans","plan_names (codes)","share_common"])
        .sort_values("claims", ascending=False).to_string(index=False))
