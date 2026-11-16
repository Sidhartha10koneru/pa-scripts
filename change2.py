ls -d v6_impact*/ ; for d in v6_impact*/; do echo "$d $(($(wc -l < $d/P_list_thr2.csv)-1)) codes, $(($(wc -l < $d/R2_payor_exceptions.csv)-1)) exceptions"; done

python compare_v8d.py base=exp_v8d_grp_s42 l6pax=exp_v8d_l6pax_s42 l7spax=exp_v8d_l7spax_s42 --base base --out v8d_compare \
  --v8-path "s3://cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v8_draft/ai_prior_auth_09282026_v3_{year}_std.parquet"
