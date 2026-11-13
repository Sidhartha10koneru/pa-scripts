PATH = "cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v8_draft/ai_prior_auth_09282026_v3_{year}_std.parquet"
COLS = ["CSN", "FINAL_CPT", "FINAL_MODIFIERS", "PLACE_OF_SERVICE_CODE", "SERVICEDATE", "FILING_ORDER_C",
        "PAYOR", "PLAN", "GROUP_GUID"]
TEST_START, TEST_END = pd.Timestamp("2026-05-01"), pd.Timestamp("2026-08-01")
