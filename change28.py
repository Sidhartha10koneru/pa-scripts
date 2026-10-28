import json, numpy as np, pandas as pd, lightgbm as lgb
from pa_base_adapter import apply_categories

feats = ["FINAL_CPT","FINAL_MODIFIERS","PLACE_OF_SERVICE_CODE","PAYOR_CLEAN","PAYOR_PLAN_CLEAN","RULE_PLAN_SHARE"]
base = lgb.Booster(model_file="exp_d2_cells/base_frozen.txt")
cats = json.load(open("exp_d2_cells/base_frozen_categories.json"))
w    = pd.read_csv("exp_d2_cells/weights_cpt_plan_k20.csv", index_col="cell")      # Feb–Apr correction table
t    = pd.read_parquet("exp_d1_s30/test_preds.parquet")                             # May–Jul test lines

sig = lambda z: 1 / (1 + np.exp(-z))
t["cell"]        = t["FINAL_CPT"].astype(str) + "|" + t["PAYOR_PLAN_CLEAN"].astype(str)
t["raw_frozen"]  = base.predict(apply_categories(t, cats, feats), raw_score=True)
t["weight"]      = t["cell"].map(w["weight"]).fillna(0)
t["p_frozen"]    = sig(t.raw_frozen)
t["p_corrected"] = sig(t.raw_frozen + t.weight)

# one row per cell: Feb–Apr learning + May–Jul result
sheet = (t.groupby("cell").agg(test_lines=("y","size"), test_actual_pa=("y","mean"),
                               test_frozen_avg=("p_frozen","mean"), test_corrected_avg=("p_corrected","mean"),
                               weight=("weight","first"))
           .join(w[["n","obs","pred"]].rename(columns={"n":"febapr_lines","obs":"febapr_actual_pa","pred":"febapr_frozen_avg"}))
           .sort_values("test_lines", ascending=False))
sheet.round(4).to_csv("cell_weights_sheet.csv")
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
print(sheet.head(25).round(3).to_string())

# a few individual claim lines, before vs after
print(t[["cell","FINAL_MODIFIERS","PLACE_OF_SERVICE_CODE","y","p_frozen","weight","p_corrected"]]
      .sample(15, random_state=1).round(3).to_string())
