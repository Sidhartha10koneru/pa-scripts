def views(x, name):
    x = x.sort_values(["PAT_ID", "SVC"]).copy()
    x["prior_pa"] = x.groupby("PAT_ID").y.transform(lambda q: q.shift().cummax()).fillna(0)
    print(f"\n===== {name}")
    print("by month:\n", x.groupby(x.SVC.dt.to_period("M")).y.agg(lines="size", pa="sum", rate="mean").to_string())
    print("by patient history:\n", x.groupby("prior_pa").y.agg(lines="size", pa="sum", rate="mean").to_string())
    t1 = x[x.y == 1]
    src = np.select([t1.LEVEL1_LABEL.astype(str) == "TRUE", t1.LEVEL2_LABEL.astype(str) == "TRUE"],
                    ["auth number", "denial code"], "other remit row")
    print("TRUE source:\n", pd.Series(src).value_counts().to_string())

def load_cell(cpt, plan, mod, pos="11", years=(2025, 2026)):
    v = pd.concat([ds.dataset(pb.BASE_PATH.format(year=yy).replace("s3://", ""), filesystem=fs, format="parquet")
                   .to_table(filter=ds.field("FINAL_CPT") == cpt).to_pandas() for yy in years], ignore_index=True)
    v = pb.clean_names(v.loc[:, ~v.columns.duplicated()])
    m = v.FINAL_MODIFIERS.astype(object).fillna("").astype(str).str.strip()
    x = v[(v.PAYOR_PLAN_CLEAN == plan) & (v.PLACE_OF_SERVICE_CODE.astype(str) == pos) & (m == mod)
          & v.GROUP_LABEL.astype(str).str.upper().isin(["TRUE", "FALSE"])].copy()
    x["y"] = (x.GROUP_LABEL.astype(str).str.upper() == "TRUE").astype(int)
    x["SVC"] = pd.to_datetime(x.SERVICEDATE, errors="coerce")
    return x

views(load_cell("99214", "EMBLEM HEALTH/HIP | EMBLEM HEALTH/HIP", ""), "HIP 99214")
views(load_cell("97112", "EMBLEM HEALTH / GHI | NYCE PPO PLAN", "GP"), "GHI PT 97112")
