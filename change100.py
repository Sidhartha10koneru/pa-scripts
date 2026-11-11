V8D="s3://cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v8_draft/ai_prior_auth_09282026_v3_{year}_std.parquet"
nohup bash -c "python compare_v8d.py base=exp_v8d_grp_s42 l6pax=exp_v8d_l6pax_s42 l7spax=exp_v8d_l7spax_s42 l7pax=exp_v8d_l7pax_s42 \
   --base base --out v8d_compare_s42 --v8-path '$V8D' > run_v8d_compare_s42.log 2>&1; \
 python compare_v8d.py base=exp_exclude_s7 l6pax=exp_v8d_l6pax_s7 l7spax=exp_v8d_l7spax_s7 l7pax=exp_v8d_l7pax_s7 \
   --base base --out v8d_compare_s7 --v8-path '$V8D' > run_v8d_compare_s7.log 2>&1" > /dev/null 2>&1 &
