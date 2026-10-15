import pandas as pd, pyarrow.fs as pafs
s3 = pafs.S3FileSystem()
base = "cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v4/ai_prior_auth_09282026_v3_{year}_std.parquet"

parts = []
for y in [2022, 2023, 2024, 2025, 2026]:
    d = pd.read_parquet(base.format(year=y), columns=["CSN", "PAYOR", "PLAN", "GROUP_LABEL"], filesystem=s3)
    d = d[d.GROUP_LABEL.astype(str).isin(["TRUE", "FALSE"])]
    d["is_pa"] = d.GROUP_LABEL.astype(str).eq("TRUE")
    # one row per encounter x payor x plan: PA if any line is TRUE
    e = d.groupby(["CSN", "PAYOR", "PLAN"], observed=True).is_pa.max().reset_index()
    t = e.groupby(["PAYOR", "PLAN"], observed=True).agg(encounters=("is_pa", "size"), pa_encounters=("is_pa", "sum")).reset_index()
    t["year"] = y
    parts.append(t)
    del d, e

pp = pd.concat(parts)
tot = pp.groupby(["PAYOR", "PLAN"], observed=True)[["encounters", "pa_encounters"]].sum()
tot["pa_rate_%"] = (100 * tot.pa_encounters / tot.encounters).round(2)
tot["share_of_all_pa_%"] = (100 * tot.pa_encounters / tot.pa_encounters.sum()).round(2)
tot = tot.sort_values("pa_encounters", ascending=False)
tot["cum_share_%"] = tot["share_of_all_pa_%"].cumsum().round(1)

print("payor+plan combinations:", len(tot), "| with >=50 PA encounters:", (tot.pa_encounters >= 50).sum())
print(tot.head(30).to_string())

by_year = pp.pivot_table(index=["PAYOR", "PLAN"], columns="year", values="pa_encounters", aggfunc="sum", fill_value=0)
with pd.ExcelWriter("pa_by_payor_plan.xlsx") as w:
    tot.to_excel(w, sheet_name="all_years")
    by_year.loc[tot.index].to_excel(w, sheet_name="pa_by_year")
  
