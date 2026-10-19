import pandas as pd, pyarrow.fs as pafs
s3 = pafs.S3FileSystem()
cl = pd.read_parquet("cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v4/ai_prior_auth_09282026_v3_2026_std.parquet",
                     columns=["BUCKET_PAYOR_ID", "BUCKET_PLAN_ID"], filesystem=s3).astype(str).drop_duplicates()
p = pd.read_csv("Prod_Globals/payerProfile_Map.csv", dtype=str)
m = pd.read_csv("Prod_Globals/PayerIdPlanId_Map.csv", dtype=str)
claim_ids = {"BUCKET_PAYOR_ID": set(cl.BUCKET_PAYOR_ID.str.strip()), "BUCKET_PLAN_ID": set(cl.BUCKET_PLAN_ID.str.strip())}
rule_ids = {c: set(p[c].dropna().str.strip()) for c in ["RH_PAYOR_ID", "PAYOR_SEQ_ID", "PAYOR_XREF_SEQ_ID", "PAYOR_XREF_ID", "PAYOR_ALIAS", "GLOBAL_PAYER_ID"]}
rule_ids["PLAN_ID"] = set(m.PLAN_ID.str.strip())
for a, A in claim_ids.items():
    for b, B in rule_ids.items():
        print(f"{a:16} vs {b:18}: {len(A & B)} shared values")
