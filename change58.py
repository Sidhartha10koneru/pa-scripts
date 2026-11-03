nohup python -u cell_separability.py --cells exp_nofilter_2022/fn_mixed_cells_recall80.csv --cover 0.5 \
  --primary-only --min-pa-eval 20 \
  --drop-features month CHARGEAMOUNT POLICY_EFFECTIVE_FROM ETHNICITY RACE SEX RELATIONSHIP_TO_SUBSCRIBER \
  --out sep_mixed_v3 > run_sep_mixed_v3.log 2>&1 &
