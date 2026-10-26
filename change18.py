import pa_base_adapter as pb
s2 = pd.read_parquet(path_v4_2026, columns=["FINAL_CPT","PAYOR","PLAN","GROUP_LABEL","N_TRUE","N_FALSE"],
                     filesystem=pafs.S3FileSystem())
s2["flipped"] = (s2.GROUP_LABEL.astype(str)=="FALSE") & (s2.N_FALSE==0) & (s2.N_TRUE==0)
s2 = s2[s2.flipped].copy()
s2 = pb.add_rules_feature(s2.rename(columns={"FINAL_CPT": pb.CPT}),
     "/home/sagemaker-user/user-default-efs/ksidhar/v3data_eda_analysis/Prd_rules_analysis/payor_master_review.xlsx",
     "rules_analysis/all_rules.parquet")
print(s2.RULE_PLAN_SHARE.describe())
print("flipped lines whose CPT is on the payer's PA list:", f"{(s2.RULE_PLAN_SHARE > 0).sum():,}")
print(s2[s2.RULE_PLAN_SHARE > 0].groupby([pb.CPT,"PAYOR"], observed=True).size().sort_values(ascending=False).head(20).to_string())
