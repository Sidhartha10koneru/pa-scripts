fb = pd.read_csv("exp_nofilter_2022/fn_buckets_recall80_min20.csv", dtype={"FINAL_CPT": str})
CUT = 0.421

# Bucket 2 (rare): is the cell new, or would pooling by CPT have caught it?
r2 = fb[fb.bucket.str.startswith("2")]
print(len(r2), "FN | cell never seen in training:", int((r2.train_n == 0).sum()))
print(r2.train_n.describe().round(1))
print("CPT-level training rate >= cut-off (pooling by CPT would score it high):",
      int((r2.train_cpt_pa_rate >= CUT).sum()))
print(r2.groupby("FINAL_CPT").size().sort_values(ascending=False).head(15))

# Bucket 5 (model miss): high training rate, still scored low - small cells blended in leaves?
r5 = fb[fb.bucket.str.startswith("5")]
print(len(r5), "FN"); print(r5[["train_n", "train_pa_rate", "score"]].describe().round(3))
print("cells with < 100 training lines (below the model's leaf minimum):", int((r5.train_n < 100).sum()))
