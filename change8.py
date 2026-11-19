nohup python -u build_v9_optc.py \
  --out-path "s3://cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v9_draft/ai_prior_auth_09282026_v3_{year}_std.parquet" \
  > run_build_v9.log 2>&1 &
tail -f run_build_v9.log
