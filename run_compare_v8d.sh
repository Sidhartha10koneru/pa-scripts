#!/usr/bin/env bash
# run_compare_v8d.sh - fair comparison of the v8-draft label runs, seed 42 then seed 7.
#   nohup bash run_compare_v8d.sh > /dev/null 2>&1 &
# Logs: run_v8d_compare_s42.log, run_v8d_compare_s7.log | outputs: v8d_compare_s42/, v8d_compare_s7/
set -u
V8D="s3://cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v8_draft/ai_prior_auth_09282026_v3_{year}_std.parquet"
[ -f compare_v8d.py ] || { echo "compare_v8d.py not in $(pwd) - upload it first"; exit 1; }

python -u compare_v8d.py base=exp_v8d_grp_s42 l6pax=exp_v8d_l6pax_s42 l7spax=exp_v8d_l7spax_s42 l7pax=exp_v8d_l7pax_s42 \
  --base base --out v8d_compare_s42 --v8-path "$V8D" > run_v8d_compare_s42.log 2>&1

# seed 7: no v8d GROUP_LABEL run with seed 7; exp_exclude_s7 is the same model (run 0 showed v8 files reproduce v4 exactly)
python -u compare_v8d.py base=exp_exclude_s7 l6pax=exp_v8d_l6pax_s7 l7spax=exp_v8d_l7spax_s7 l7pax=exp_v8d_l7pax_s7 \
  --base base --out v8d_compare_s7 --v8-path "$V8D" > run_v8d_compare_s7.log 2>&1
echo "done $(date '+%F %T')" >> run_v8d_compare_s7.log
