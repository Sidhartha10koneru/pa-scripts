nohup python -u build_v7_1.py --years 2022 2023 2024 2025 2026 --impact-dir v6_impact_allyears --manifest v8_draft_manifest.json --out-path "s3://cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v8_draft/ai_prior_auth_09282026_v3_{year}_std.parquet" > run_build_v8_draft.log 2>&1 &


aws s3 cp v8_draft_manifest.json s3://cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v8_draft/v8_draft_manifest.json
