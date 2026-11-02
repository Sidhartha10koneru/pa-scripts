# training cell = all 5 model features
key = ["FINAL_CPT", "FINAL_MODIFIERS", "PLACE_OF_SERVICE_CODE", "PAYOR_PLAN_CLEAN"]
tr = d[d.SVC_DATE < "2026-05-01"].copy()
for c in key:
    tr[c] = tr[c].astype(str)
cell = tr.groupby(key).y.agg(train_n="size", train_pa_rate="mean")
cpt_rate = tr.groupby("FINAL_CPT").y.mean().rename("train_cpt_pa_rate")

# test: service = CSN + CPT (unchanged); mod / POS / plan from its highest-scoring line
t = pd.read_parquet("exp_nofilter_2022/test_preds_with_guid.parquet")
for c in key:
    t[c] = t[c].astype(str)
s = t.groupby(["CSN", "FINAL_CPT"]).agg(y=("y", "max"), score=("p_lgb_full_2022", "max")).reset_index()
top = t.loc[t.groupby(["CSN", "FINAL_CPT"]).p_lgb_full_2022.idxmax(), ["CSN"] + key]
s = s.merge(top, on=["CSN", "FINAL_CPT"])

cut = 0.421                                            # 80%-recall cut-off of the 2022 model
fn = s[(s.y == 1) & (s.score < cut)].copy()
print("FN services:", len(fn))                         # expect ~10,451
fn = fn.join(cell, on=key).join(cpt_rate, on="FINAL_CPT")
fn["train_n"] = fn.train_n.fillna(0)
