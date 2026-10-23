# ==========================================================
# PA MODEL v0 - BASE + ADAPTER - RUN SHEET
# Folder: ~/user-default-efs/ksidhar/Global_prd_rules
# Script: pa_base_adapter.py  (latest version from chat)
# ==========================================================

# --- 0. Go to the folder and check the script is the latest version
cd ~/user-default-efs/ksidhar/Global_prd_rules
grep -n "PAYOR_CLEAN\|FINAL_CPT\|operating_points" pa_base_adapter.py
#   -> should print lines for all three; if not, upload the new file again

# --- 1. Check lightgbm is installed
pip install lightgbm

# --- 2. Check the claims columns the script needs
python -c "
import pyarrow.parquet as pq, pyarrow.fs as f
names = pq.read_schema('cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v4/ai_prior_auth_09282026_v3_2025_std.parquet', filesystem=f.S3FileSystem()).names
need = ['CSN','GROUP_LABEL','SERVICEDATE','FILING_ORDER_C','FINAL_CPT','FINAL_MODIFIERS','PLACE_OF_SERVICE_CODE','PAYOR','PLAN','LINE_OF_BUSINESS']
print('missing:', [c for c in need if c not in names])
"
#   -> should print: missing: []

# ==========================================================
# TRAIN BASE ONCE, THEN ADD ADAPTERS ON TOP OF THE SAME BASE
# ==========================================================

# --- A. Train the base (+ first adapter: rules). This creates the base we keep.
python pa_base_adapter.py --train-years 2025 --test-year 2026 --sample 0.1 --full --rules-master /home/sagemaker-user/user-default-efs/ksidhar/v3data_eda_analysis/Prd_rules_analysis/payor_master_review.xlsx --rules-parquet rules_analysis/all_rules.parquet --out pa_model_v0_rules 2>&1 | tee run_rules.log

# --- B. Reuse that base, adapter = LINE_OF_BUSINESS (base NOT retrained)
python pa_base_adapter.py --train-years 2025 --test-year 2026 --sample 0.1 --full --base-model pa_model_v0_rules/base_model.txt --new-feats LINE_OF_BUSINESS --out pa_model_v0_lob 2>&1 | tee run_lob.log

# --- C. Reuse the same base, adapter = context features
python pa_base_adapter.py --train-years 2025 --test-year 2026 --sample 0.1 --full --base-model pa_model_v0_rules/base_model.txt --new-feats Primary_DX SPECIALTY PRODUCT_NAME VISITTYPE --out pa_model_v0_ctx 2>&1 | tee run_ctx.log

# --- Proof the base was reused, not retrained
grep -E "Reusing|BASE loaded|unchanged" run_lob.log
grep -E "base_reused|base_model_source" pa_model_v0_lob/run_info.json
#   -> "BASE loaded (not retrained)", "base_reused": true
#   -> base row in metrics.csv is identical across all runs

# --- 7. Show results again without re-running (change folder name as needed)
grep -E "lines,|rules feature|unchanged" run_rules.log
cat pa_model_v0_rules/metrics.csv
cat pa_model_v0_rules/operating_points.csv
cat pa_model_v0_rules/feature_importance.csv
cat pa_model_v0_rules/run_info.json
ls pa_model_v0_rules
#   base_model.txt      = frozen base model
#   adapter_model.txt   = adapter trained on top of base
#   full_model.txt      = full retrain, for comparison

# --- 8. Nicer table view in the terminal
python -c "
import pandas as pd; pd.set_option('display.width',250)
d='pa_model_v0_rules'
print(pd.read_csv(f'{d}/metrics.csv', index_col=0).round(4))
o=pd.read_csv(f'{d}/operating_points.csv'); print(o[o.unit=='encounter'].round(4).to_string(index=False))
"

# ==========================================================
# WHAT EACH OPTION MEANS
#   --train-years   years used to train (e.g. 2024 2025)
#   --test-year     year held out to test (2026)
#   --sample        share of encounters kept (0.1 = 10%, saves memory)
#   --full          also train a model from scratch with all features, to compare
#   --new-feats     extra claim column(s) for the adapter
#   --rules-master  name-based payor mapping (claims PAYOR/PLAN -> rules payor code)
#   --rules-parquet latest rules (686 plans) combined into one file
#   --out           output folder
# ==========================================================
