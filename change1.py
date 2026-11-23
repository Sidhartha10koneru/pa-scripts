python confusion_v9.py base=exp_v8d_grp_s42 l6pax=exp_v8d_l6pax_s42 l6paxc=exp_v9d_l6paxc_s42 --base base --recall 0.85 --out v8_baseline_85.csv

python -c "import pyarrow.parquet as pq, pyarrow.fs as f; s=pq.read_schema('cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v9_draft/ai_prior_auth_09282026_v3_2026_std.parquet', filesystem=f.S3FileSystem()); print(len(s.names)); print(s.names)"
