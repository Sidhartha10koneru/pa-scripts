nohup bash -c 'bash run_v8d.sh 2 3 4 5 6 > run_v8d_rest.log 2>&1; \
python compare_v8d.py base=exp_v8d_grp_s42 l7spax=exp_v8d_l7spax_s42 l6pax=exp_v8d_l6pax_s42 l7pax=exp_v8d_l7pax_s42 \
  --base base --out v8d_compare_s42 \
  --v8-path "s3://cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v8_draft/ai_prior_auth_09282026_v3_{year}_std.parquet" \
  > run_v8d_compare.log 2>&1' > /dev/null 2>&1 &

grep -E "^pa-|AUC_PR|FP@80|coverage|FAILED" run_v8d_rest.log
cat run_v8d_compare.log

   ps aux | grep -E "pa_experiments|run_v8d|compare_v8d" | grep -v grep
   sleep 60; head -5 run_v8d_rest.log
