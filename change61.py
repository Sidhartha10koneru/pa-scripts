f["share_type"] = f.apply(share_type, axis=1)
f.loc[f.FINAL_CPT.str.match(r"^(97\d{3}|9894[0-2]|92507|92526)$") & f.share_type.str.startswith("visit"), "share_type"] = "therapy episode"
print(pd.crosstab(f.share_type, f.t1_source, margins=True))
u = f[(f.share_type == "not shared") & f.cause.str.startswith("UNEXPLAINED")]
print(u.t1_source.value_counts())
print(u.groupby(["FINAL_CPT", "PAYOR_PLAN_CLEAN"]).size().sort_values(ascending=False).head(15))
print(u[["cell_fit_rate", "cell_test_rate"]].astype(float).describe().round(3))
