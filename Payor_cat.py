import re, pandas as pd, pyarrow.fs as pafs
s3 = pafs.S3FileSystem()
base = "cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v4/ai_prior_auth_09282026_v3_{year}_std.parquet"
cols = ["CSN", "BUCKET_PAYOR_ID", "BUCKET_PAYOR_NAME", "BUCKET_PLAN_ID", "PAYOR", "PLAN",
        "LINE_OF_BUSINESS", "PRODUCT_NAME", "GROUP_LABEL"]
keys = ["BUCKET_PAYOR_ID", "BUCKET_PLAN_ID"]

enc, lob, prod, names = [], [], [], []
for y in [2024, 2025, 2026]:
    d = pd.read_parquet(base.format(year=y), columns=cols, filesystem=s3)
    for c in cols:
        d[c] = d[c].astype(str).str.strip().replace({"": None, "nan": None, "None": None})
    d["is_pa"] = d.GROUP_LABEL.eq("TRUE")
    lab = d[d.GROUP_LABEL.isin(["TRUE", "FALSE"])]
    e = lab.groupby(keys + ["CSN"]).is_pa.max().reset_index()
    enc.append(e.groupby(keys).agg(encounters=("is_pa", "size"), pa_encounters=("is_pa", "sum")))
    lob.append(d.dropna(subset=["LINE_OF_BUSINESS"]).groupby(keys + ["LINE_OF_BUSINESS"]).size())
    prod.append(d.dropna(subset=["PRODUCT_NAME"]).groupby(keys + ["PRODUCT_NAME"]).size())
    names.append(d.groupby(keys + ["BUCKET_PAYOR_NAME", "PAYOR", "PLAN"]).size())
    del d, lab, e

enc = pd.concat(enc).groupby(level=keys).sum()

def top(parts, col):
    s = pd.concat(parts).groupby(level=keys + [col]).sum().rename("n").reset_index()
    s["share"] = s.n / s.groupby(keys).n.transform("sum")
    s = s.sort_values("n", ascending=False).drop_duplicates(keys)
    return s.set_index(keys)[[col, "share"]].rename(columns={"share": col + "_share"})

nm = pd.concat(names).groupby(level=keys + ["BUCKET_PAYOR_NAME", "PAYOR", "PLAN"]).sum().rename("n").reset_index()
nm = nm.sort_values("n", ascending=False).drop_duplicates(keys).set_index(keys)[["BUCKET_PAYOR_NAME", "PAYOR", "PLAN"]]
out = nm.join(enc).join(top(lob, "LINE_OF_BUSINESS")).join(top(prod, "PRODUCT_NAME"))

txt = (out.BUCKET_PAYOR_NAME.fillna("") + " " + out.PAYOR.fillna("") + " " + out.PLAN.fillna("") + " " + out.PRODUCT_NAME.fillna("")).str.upper()
rules = [(r"SELF.?PAY", "Self-pay"), (r"WORKER|COMP\b|NO.?FAULT|AUTO|TPL|LIABILITY|ONE CALL", "Workers comp / auto / TPL"),
         (r"TRICARE|CHAMPVA|\bVA\b|VETERAN|TRIWEST", "Tricare / VA"),
         (r"SUPPLEMENT|MEDIGAP", "Medicare supplement"),
         (r"ADVANTAGE|MEDICARE REPL|\bDUAL\b|\bD-?SNP\b|\bSNP\b|MEDICARE HMO|MEDICARE PPO", "Medicare Advantage"),
         (r"MEDICAID|HEALTHPLUS|COMMUNITY PLAN|ESSENTIAL|CHILD HEALTH PLUS|\bCHP\b", "Medicaid managed care"),
         (r"\bMEDICARE\b|RAILROAD", "Original Medicare"),
         (r"\bQHP\b|MARKETPLACE|EXCHANGE", "Marketplace (QHP)")]
out["keyword_segment"] = None
for pat, seg in rules:
    m = out.keyword_segment.isna() & txt.str.contains(pat, regex=True)
    out.loc[m, "keyword_segment"] = seg

lob_map = {"Commercial": "Commercial", "Blue Cross": "Commercial", "Medicaid Replacement": "Medicaid managed care",
           "Medicaid": "Medicaid", "Medicare Replacement": "Medicare Advantage", "Medicare": "Original Medicare",
           "Qualified Health Plan": "Marketplace (QHP)", "Tricare": "Tricare / VA", "Self-Pay": "Self-pay",
           "Worker's Comp": "Workers comp / auto / TPL", "TPL": "Workers comp / auto / TPL"}
out["lob_segment"] = out.LINE_OF_BUSINESS.map(lob_map)
out["proposed_segment"] = out.lob_segment.fillna(out.keyword_segment).fillna("Commercial (default)")
out["conflict"] = out.lob_segment.notna() & out.keyword_segment.notna() & (out.lob_segment != out.keyword_segment) \
                  & ~((out.lob_segment == "Medicaid") & (out.keyword_segment == "Medicaid managed care"))
out["source"] = out.lob_segment.notna().map({True: "LOB", False: "keyword"}).where(out.keyword_segment.notna() | out.lob_segment.notna(), "default")
out["reviewed_segment"] = ""          # fill by hand for top rows / conflicts
out["payer_family"] = ""              # fill by hand: UHC, Aetna, Anthem/Elevance, ...
out = out.sort_values("pa_encounters", ascending=False)
out["cum_pa_share"] = (out.pa_encounters.cumsum() / out.pa_encounters.sum()).round(3)
out.to_excel("payor_plan_categories.xlsx")

print("plans:", len(out), "| conflicts:", int(out.conflict.sum()), "| default:", int((out.source == "default").sum()))
print(out.groupby("proposed_segment")[["encounters", "pa_encounters"]].sum().assign(
      pa_rate=lambda t: (100 * t.pa_encounters / t.encounters).round(2)).sort_values("pa_encounters", ascending=False))
