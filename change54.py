f = pd.read_csv("rc_mixed/fn_causes.csv", dtype=str)
p5 = f[(f.t1_source == "auth number") & f.t2_shared_with.fillna("").eq("")].copy()
print(len(p5)); print(p5.cause.value_counts())
print(p5.groupby(["FINAL_CPT", "PAYOR_PLAN_CLEAN"]).size().sort_values(ascending=False).head(12))
cols = [c for c in ["FINAL_CPT", "PAYOR_PLAN_CLEAN", "cause", "t4_attr", "t4_value", "t4_rate", "t3_auth_age_days",
                    "cell_fit_rate", "cell_test_rate"] if c in p5]
print(p5[cols].head(15).to_string())
