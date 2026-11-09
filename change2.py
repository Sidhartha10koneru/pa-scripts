nohup python -u v6_label_impact.py --years 2022 2023 2024 2025 2026 --rate-basis amb_false --thresholds 0.01 0.02 0.05 --test-start 2026-08-01 --out v6_impact_allyears > run_v6_impact_allyears.log 2>&1 &

import pandas as pd
a = set(pd.read_csv("v6_impact_owner/P_list_thr2.csv", dtype=str).FINAL_CPT)
b = set(pd.read_csv("v6_impact_allyears/P_list_thr2.csv", dtype=str).FINAL_CPT)
ea = pd.read_csv("v6_impact_owner/R2_payor_exceptions.csv", dtype=str); eb = pd.read_csv("v6_impact_allyears/R2_payor_exceptions.csv", dtype=str)
print(f"list: current {len(a)} | all-years {len(b)} | only current {sorted(a-b)} | only all-years {sorted(b-a)}")
print(f"exceptions: current {len(ea)} | all-years {len(eb)}")


python build_v7_1.py --years 2022 2023 2024 2025 2026 --impact-dir v6_impact_allyears --dry-run
nohup python -u build_v7_1.py --years 2022 2023 2024 2025 2026 --impact-dir v6_impact_allyears --manifest v8_draft_manifest.json --out-path "s3://cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v8_draft/ai_prior_auth_09282026_v3_{year}_std.parquet" > run_build_v8_draft.log 2>&1 &
