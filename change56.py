import pandas as pd

path = "exp_nofilter_2022/test_preds_with_guid.parquet"
model, cut = "p_lgb_full_2022", 0.80          # 2022-2026 model, best_thr at service level

t = pd.read_parquet(path)
t["GROUP_GUID"] = t.GROUP_GUID.astype(str)

# 1) service level: PA if any line TRUE, score = highest line score
s = t.groupby(["CSN", "FINAL_CPT"]).agg(y=("y", "max"), score=(model, "max"),
                                        n_lines=("GROUP_GUID", "size")).reset_index()
fn = s[(s.y == 1) & (s.score < cut)]
print("FN services:", len(fn))                 # expect 18,070

# 2) all lines (GUIDs) behind those services
lines = t.merge(fn[["CSN", "FINAL_CPT", "score"]].rename(columns={"score": "service_score"}),
                on=["CSN", "FINAL_CPT"])
lines = lines[["GROUP_GUID", "CSN", "FINAL_CPT", "FINAL_MODIFIERS", "SERVICEDATE", "FILING_ORDER_C",
               "PAYOR_CLEAN", "PAYOR_PLAN_CLEAN", "PLACE_OF_SERVICE_CODE", "y", model, "service_score"]]
print("lines (GUIDs) in FN services:", len(lines), "| of which TRUE lines:", int(lines.y.sum()))

# 3) one row per FN service with its GUIDs listed
fn_svc = fn.merge(lines.groupby(["CSN", "FINAL_CPT"]).GROUP_GUID.agg(lambda x: ",".join(x)).rename("GROUP_GUIDS"),
                  on=["CSN", "FINAL_CPT"])

lines.to_parquet("exp_nofilter_2022/fn_2022_service_lines.parquet", index=False)
fn_svc.to_csv("exp_nofilter_2022/fn_2022_services.csv", index=False)

# quick view: which CPTs are missed most
print(fn.FINAL_CPT.value_counts().head(20))
# lines = the FN service lines from the previous cell
idx = lines.groupby(["CSN", "FINAL_CPT"])[model].idxmax()
fn_top = lines.loc[idx].reset_index(drop=True)
print(len(fn_top))                                  # 18,070: one GUID per FN service
fn_top.to_csv("exp_nofilter_2022/fn_2022_top_guid.csv", index=False)





