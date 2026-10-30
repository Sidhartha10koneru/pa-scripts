import json, pandas as pd, lightgbm as lgb
import pa_base_adapter as pb

t = pd.read_parquet("exp_nofilter_2025_ids/test_preds.parquet")          # same test rows, with GROUP_GUID
m = lgb.Booster(model_file="exp_nofilter_2022/models/lgb_full.txt")      # 2022+ model, already trained
cats = json.load(open("exp_nofilter_2022/models/lgb_full_categories.json"))
t["p_lgb_full_2022"] = m.predict(pb.apply_categories(t, cats, pb.BASE_FEATS))
t = t.rename(columns={"p_lgb_full": "p_lgb_full_2025"})

print(len(t), t.GROUP_GUID.nunique())                                    # expect 2182493 2182493
t.to_parquet("exp_nofilter_2022/test_preds_with_guid.parquet", index=False)


from sklearn.metrics import average_precision_score as ap
s = t.groupby(["CSN", "FINAL_CPT"]).agg(y=("y", "max"), a=("p_lgb_full_2025", "max"), b=("p_lgb_full_2022", "max"))
print(round(ap(s.y, s.a), 4), round(ap(s.y, s.b), 4))                   # expect ~0.644 and ~0.620


aws s3 cp exp_nofilter_2025_ids/test_preds.parquet \
  s3://cli-ml-data-prod/datalake/ai-prior-auth/09282026/model_outputs/benchmark_v2_test_preds.parquet
