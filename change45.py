pt = pd.concat([ds.dataset(pb.BASE_PATH.format(year=2026).replace("s3://", ""), filesystem=fs, format="parquet")
                .to_table(filter=ds.field("FINAL_CPT").isin(["97110", "97112", "97140", "97116", "97530", "97161", "97162", "97163"])).to_pandas()])
pt = pb.clean_names(pt.loc[:, ~pt.columns.duplicated()])
pt = pt[(pt.PAYOR_PLAN_CLEAN == "EMBLEM HEALTH / GHI | NYCE PPO PLAN") & pt.GROUP_LABEL.astype(str).str.upper().isin(["TRUE", "FALSE"])].copy()
pt["y"] = (pt.GROUP_LABEL.astype(str).str.upper() == "TRUE").astype(int)
pt["SVC"] = pd.to_datetime(pt.SERVICEDATE, errors="coerce")
v = pt.groupby(["PAT_ID", "SVC"]).y.max().reset_index().sort_values(["PAT_ID", "SVC"])   # one row per patient-day = PT visit
v["visit_no"] = v.groupby("PAT_ID").cumcount() + 1                                        # PT visit number in 2026
first = v[v.y == 1].groupby("PAT_ID").visit_no.min()
print("PT visit number at which the first PA appears:\n", first.value_counts().sort_index().to_string())
tot = v.groupby("PAT_ID").visit_no.max()
print("\ntotal 2026 PT visits - patients never PA:\n", tot[~tot.index.isin(first.index)].describe().round(1).to_string())
print("\ntotal 2026 PT visits - patients with PA:\n", tot[tot.index.isin(first.index)].describe().round(1).to_string())
