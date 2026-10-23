sed -i 's/CPT, MOD, POS = "CPT",/CPT, MOD, POS = "FINAL_CPT",/' pa_base_adapter1.py
grep -n 'CPT, MOD, POS' pa_base_adapter1.py

python -c "
import pyarrow.parquet as pq, pyarrow.fs as f
names = pq.read_schema('cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v4/ai_prior_auth_09282026_v3_2025_std.parquet', filesystem=f.S3FileSystem()).names
need = ['CSN','GROUP_LABEL','SERVICEDATE','FILING_ORDER_C','FINAL_CPT','FINAL_MODIFIERS','PLACE_OF_SERVICE_CODE','PAYOR','PLAN','LINE_OF_BUSINESS']
print('missing:', [c for c in need if c not in names]); print(names)
"
