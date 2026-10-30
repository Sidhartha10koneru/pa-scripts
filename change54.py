import pyarrow.parquet as pq

notes = {
 "p_lgb_full_2025": "BENCHMARK v2. LightGBM trained 2025-01-01..2026-04-30 (run exp_nofilter_2025). AUC-PR service 0.644 / visit 0.688",
 "p_lgb_full_2022": "LightGBM trained 2022-01-01..2026-04-30 (run exp_nofilter_2022). AUC-PR service 0.620 / visit 0.672",
 "y": "label from GROUP_LABEL: 1 = TRUE, 0 = FALSE",
 "test": "service dates 2026-05-01..2026-07-31; all filing orders; AMBIGUOUS lines and exact repeats excluded",
 "features": "FINAL_CPT, FINAL_MODIFIERS, PLACE_OF_SERVICE_CODE, PAYOR_CLEAN, PAYOR_PLAN_CLEAN",
 "scores": "model score 0-1, not a calibrated probability; one row per test line",
 "join": "GROUP_GUID is unique per row; join to v4 std_data on GROUP_GUID",
}
path = "exp_nofilter_2022/test_preds_with_guid.parquet"
tbl = pq.read_table(path)
tbl = tbl.replace_schema_metadata({**(tbl.schema.metadata or {}), **{k.encode(): v.encode() for k, v in notes.items()}})
pq.write_table(tbl, path)
open("exp_nofilter_2022/test_preds_with_guid_README.txt", "w").write("\n".join(f"{k}: {v}" for k, v in notes.items()))
print(open("exp_nofilter_2022/test_preds_with_guid_README.txt").read())


aws s3 cp exp_nofilter_2022/test_preds_with_guid.parquet \
  s3://cli-ml-data-prod/datalake/ai-prior-auth/09282026/model_outputs/test_preds_with_guid.parquet
aws s3 cp exp_nofilter_2022/test_preds_with_guid_README.txt \
  s3://cli-ml-data-prod/datalake/ai-prior-auth/09282026/model_outputs/test_preds_with_guid_README.txt


aws s3 ls s3://cli-ml-data-prod/datalake/ai-prior-auth/09282026/model_outputs/
