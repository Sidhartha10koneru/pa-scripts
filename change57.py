import re, numpy as np, pandas as pd, pyarrow.fs as pafs
import pa_base_adapter as pb, pa_experiments as pe

# 1) training data, same settings as the 2022 run
pb.INCLUDE_SECONDARY, pb.DEDUP_KEEP_TRUE = True, True
cols = [pb.CSN, pb.LABEL, pb.DATE, pb.FILING, "FINAL_CPT", "FINAL_MODIFIERS", "PLACE_OF_SERVICE_CODE", "PAYOR", "PLAN"]
d = pe.add_dates(pb.load([2022, 2023, 2024, 2025, 2026], cols, 1.0, pafs.S3FileSystem()))
tr = d[d.SVC_DATE < "2026-05-01"]
cell = tr.groupby(["FINAL_CPT", "PAYOR_PLAN_CLEAN"]).y.agg(train_n="size", train_pa_rate="mean")
cpt_rate = tr.groupby("FINAL_CPT").y.mean().rename("train_cpt_pa_rate")

# 2) FN services + all test lines of their visits
t = pd.read_parquet("exp_nofilter_2022/test_preds_with_guid.parquet")
s = t.groupby(["CSN", "FINAL_CPT"]).agg(y=("y", "max"), score=("p_lgb_full_2022", "max"),
                                        plan=("PAYOR_PLAN_CLEAN", "first")).reset_index()
fn = s[(s.y == 1) & (s.score < 0.80)].copy()
fn = fn.join(cell, on=["FINAL_CPT", "plan"]).join(cpt_rate, on="FINAL_CPT")
fn["train_n"] = fn.train_n.fillna(0)

def fam(c):
    c = str(c).upper()
    if re.fullmatch(r"\d{4}F", c): return "quality"
    if re.fullmatch(r"99[2-4]\d\d|G2211", c): return "E/M"
    if re.fullmatch(r"8\d{4}|36415", c): return "lab"
    if re.fullmatch(r"7\d{4}|[1-6]\d{4}|J\d{4}|97\d{3}", c): return "anchor"   # imaging, procedure, drug, PT
    return "other"

# is there another TRUE "anchor" service in the same visit? (passenger signal)
pa_svc = s[s.y == 1].assign(f=lambda x: x.FINAL_CPT.map(fam))
anchor_visits = set(pa_svc.loc[pa_svc.f == "anchor", "CSN"])
fn["family"] = fn.FINAL_CPT.map(fam)
fn["anchor_in_visit"] = fn.CSN.isin(anchor_visits) & (fn.family != "anchor")

fn["bucket"] = np.select(
    [fn.family.eq("quality") | (fn.family.isin(["E/M", "lab"]) & fn.anchor_in_visit),
     fn.train_n < 20,
     fn.train_pa_rate < 0.05,
     fn.train_pa_rate < 0.50],
    ["1 likely label noise (passenger/quality)", "2 unseen or rare cell", "3 historically non-PA cell",
     "4 mixed cell (needs features)"],
    "5 genuine model miss")

summary = fn.groupby("bucket").agg(services=("CSN", "size"), median_score=("score", "median"),
                                   median_train_rate=("train_pa_rate", "median")).assign(
    share=lambda x: (x.services / x.services.sum()).round(3))
print(summary.to_string())
for b, g in fn.groupby("bucket"):
    print(f"\n{b}: top CPTs", g.FINAL_CPT.value_counts().head(8).to_dict())
fn.to_csv("exp_nofilter_2022/fn_2022_buckets.csv", index=False)
