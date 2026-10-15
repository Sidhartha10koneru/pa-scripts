import pandas as pd, pyarrow.fs as pafs, re
s3 = pafs.S3FileSystem()
p = "cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v4/ai_prior_auth_09282026_v3_2026_std.parquet"
d = pd.read_parquet(p, columns=["CSN", "PAYOR", "PLAN", "LINE_OF_BUSINESS", "FILING_ORDER_C"], filesystem=s3)
d = d[pd.to_numeric(d.FILING_ORDER_C, errors="coerce").eq(1)].drop_duplicates(["CSN", "PAYOR", "PLAN"])
txt = (d.PAYOR.astype(str) + " " + d.PLAN.astype(str)).str.upper()
d["name_seg"] = "other/unclear"
for pat, seg in [(r"MEDICAID|HEALTHPLUS|ESSENTIAL|COMMUNITY PLAN", "Medicaid"),
                 (r"ADVANTAGE|MEDICARE REPL|DUAL|\bSNP\b", "Medicare Advantage"),
                 (r"^MEDICARE\b", "Medicare"),
                 (r"COMMERCIAL|PPO|HMO|\bEPO\b|QHP", "Commercial")]:
    d.loc[d.name_seg.eq("other/unclear") & txt.str.contains(pat, regex=True), "name_seg"] = seg
ct = pd.crosstab(d.LINE_OF_BUSINESS.fillna("(empty)"), d.name_seg)
print(ct)
print("\nrow %:\n", (100 * ct.div(ct.sum(axis=1), axis=0)).round(1))
