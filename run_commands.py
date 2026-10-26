python pa_experiments.py baselines --sample 0.3 --test-start 2026-05-01 --test-end 2026-08-01 --update-months 3 --rules-master /home/sagemaker-user/user-default-efs/ksidhar/v3data_eda_analysis/Prd_rules_analysis/payor_master_review.xlsx --rules-parquet rules_analysis/all_rules.parquet --out exp_d1_s30 2>&1 | tee run_d1_s30.log

python -c "import pandas as pd; m=pd.read_csv('exp_d1/metrics.csv',index_col=0); print(m.filter(like='_line').round(4).to_string())"
