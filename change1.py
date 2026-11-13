nohup python -u fn_buckets.py --run exp_v8d_l6pax_s42 \
  --frozen frozen/test_v1_2026-05_2026-07 --labels-from exp_v8d_l6pax_s42 > run_fn_buckets.log 2>&1 &

nohup python -u fn_buckets.py --run exp_nofilter_2022 --label GROUP_LABEL --quality-mode keep \
  --base-path "s3://cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v4/ai_prior_auth_09282026_v3_{year}_std.parquet" \
  --years 2022 2023 2024 2025 2026 --test-start 2026-05-01 \
  --frozen frozen/test_v1_2026-05_2026-07 --labels-from exp_v8d_l6pax_s42 >> run_fn_buckets.log 2>&1 &

python -u fn_buckets.py --run exp_nofilter_2022 --label GROUP_LABEL --quality-mode keep \
  --base-path "s3://cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v4/ai_prior_auth_09282026_v3_{year}_std.parquet" \
  --years 2022 2023 2024 2025 2026 --test-start 2026-05-01 >> run_fn_buckets.log 2>&1
python -c "import pandas as pd; print(pd.read_csv('exp_nofilter_2022/fn_buckets_recall80_min20.csv').bucket.value_counts().sort_index())"
