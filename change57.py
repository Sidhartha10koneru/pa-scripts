key = ["FINAL_CPT", "FINAL_MODIFIERS", "PLACE_OF_SERVICE_CODE", "PAYOR_PLAN_CLEAN"]
m = fn[fn.bucket.str.startswith("4")]

# 1) mixed cells behind the missed services, biggest first
cells = (m.groupby(key)
          .agg(fn_services=("CSN", "size"), train_n=("train_n", "first"),
               train_pa_rate=("train_pa_rate", "first"), median_score=("score", "median"))
          .sort_values("fn_services", ascending=False))

# 2) how the same cells look in the test set overall (all services, not only FN)
s_all = s.copy()
for c in key:
    s_all[c] = s_all[c].astype(str)
test_cell = s_all.groupby(key).agg(test_services=("y", "size"), test_pa=("y", "sum"))
cells = cells.join(test_cell)
cells["test_pa_rate"] = cells.test_pa / cells.test_services
cells["cum_share_of_bucket4"] = cells.fn_services.cumsum() / cells.fn_services.sum()

pd.set_option("display.width", 250)
print(len(cells), "mixed cells behind", len(m), "missed services")
print(cells.head(30).round(3).to_string())
cells.to_csv("exp_nofilter_2022/fn_2022_mixed_cells.csv")
