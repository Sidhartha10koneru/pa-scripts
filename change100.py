nohup python -u build_pa_history.py --out-path "s3://cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v11_hist_draft/ai_prior_auth_09282026_v3_{year}_std.parquet" > run_build_hist.log 2>&1 &

python hist_check.py --run exp_feat_core_s42 --lag-days 30

BASE_PATH="s3://cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v11_hist_draft/ai_prior_auth_09282026_v3_{year}_std.parquet" nohup bash run_feats.sh 7 > run_hist.log 2>&1 &
