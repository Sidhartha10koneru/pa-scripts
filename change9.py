python build_v9_optc.py --dry-run

nohup python -u build_v9_optc.py --out-path "s3://<agreed>/std_data_v9_draft/ai_prior_auth_09282026_v3_{year}_std.parquet" > run_build_v9.log 2>&1 &

export V9D="s3://<agreed>/std_data_v9_draft/ai_prior_auth_09282026_v3_{year}_std.parquet"
nohup bash run_v9d.sh 0 1 > run_v9d.log 2>&1 &

nohup bash run_v9d.sh compare > run_v9d_compare.log 2>&1 &
