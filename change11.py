nohup bash -c "
export BASE_PATH='s3://cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v11_hist_draft/ai_prior_auth_09282026_v3_{year}_std.parquet'
SEED=7 bash run_feats.sh 7
python full_review.py --run exp_feat_hist_s42 --base exp_feat_core_s42 --out full_review_hist
" > run_hist_next.log 2>&1 &


bash run_feats.sh score > /dev/null
python -c "import pandas as pd; d=pd.read_csv('feats_confusion.csv'); print(d[(d['answer key']=='agreed')&(d['target recall']=='85%')].to_string(index=False))"
cat full_review_hist/summary.txt


nohup python -u build_pa_history.py --lag-days 30 --manifest hist_lag30_manifest.json \
  --out-path "s3://cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v11b_hist_lag30_draft/ai_prior_auth_09282026_v3_{year}_std.parquet" \
  > run_build_hist_lag30.log 2>&1 &


BASE_PATH="s3://cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v11b_hist_lag30_draft/ai_prior_auth_09282026_v3_{year}_std.parquet" RUN_TAG=lag30 nohup bash run_feats.sh 7 > run_hist_lag30.log 2>&1 &

