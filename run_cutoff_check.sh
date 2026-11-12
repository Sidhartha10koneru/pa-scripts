#!/usr/bin/env bash
# run_cutoff_check.sh - cut-off chosen on validation, achieved on test: latest model + benchmark, both seeds.
#   nohup bash run_cutoff_check.sh > run_cutoff_check.log 2>&1 &
# One run at a time (each reloads the data, no training). Results: cutoff_check/<run>.csv
V8D="s3://cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v8_draft/ai_prior_auth_09282026_v3_{year}_std.parquet"
V4="s3://cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v4/ai_prior_auth_09282026_v3_{year}_std.parquet"
python -u cutoff_check.py --run exp_v8d_l6pax_s42 --seed 42 --label LABEL_V6_PAX --quality-mode exclude --base-path "$V8D"
python -u cutoff_check.py --run exp_v8d_l6pax_s7  --seed 7  --label LABEL_V6_PAX --quality-mode exclude --base-path "$V8D"
python -u cutoff_check.py --run exp_nofilter_2022 --seed 42 --label GROUP_LABEL  --quality-mode keep    --base-path "$V4"
python -u cutoff_check.py --run exp_keep_s7       --seed 7  --label GROUP_LABEL  --quality-mode keep    --base-path "$V4"
echo "done $(date '+%F %T')"
