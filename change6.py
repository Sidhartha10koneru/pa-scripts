nohup python -u build_v7.py --years 2022 2023 2024 2025 2026 --out-path "s3://cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v7/ai_prior_auth_09282026_v3_{year}_std.parquet" > run_build_v7.log 2>&1 &

   aws s3 cp v7_manifest.json s3://cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v7/v7_manifest.json
