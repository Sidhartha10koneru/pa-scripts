import json, numpy as np, pandas as pd, lightgbm as lgb
from pa_base_adapter import apply_categories

feats = ["FINAL_CPT","FINAL_MODIFIERS","PLACE_OF_SERVICE_CODE","PAYOR_CLEAN","PAYOR_PLAN_CLEAN","RULE_PLAN_SHARE"]
m    = lgb.Booster(model_file="exp_d1_s30/models/lgb_rules_full.txt")
cats = json.load(open("exp_d1_s30/models/lgb_rules_full_categories.json"))
t    = pd.read_parquet("exp_d1_s30/test_preds.parquet")       # May–Jul test lines (has all 6 features)

# pick examples: 3 high-scoring PA lines and 3 low-scoring non-PA lines
ex = pd.concat([t[t.y == 1].nlargest(3, "p_lgb_rules_full"), t[t.y == 0].nsmallest(3, "p_lgb_rules_full")])
X  = apply_categories(ex, cats, feats)

contrib = pd.DataFrame(m.predict(X, pred_contrib=True), columns=feats + ["average"], index=ex.index)
contrib["total_logodds"] = contrib.sum(axis=1)
contrib["score"] = 1 / (1 + np.exp(-contrib.total_logodds))
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 40)
print(ex[feats + ["y"]].to_string())
print(contrib.round(2).to_string())

# what-if: take the first example and set the rules feature to 0 (code not on payer's list)
w = X.iloc[[0]].copy(); w["RULE_PLAN_SHARE"] = 0.0
print("what-if rules=0:", round(m.predict(X.iloc[[0]])[0], 3), "->", round(m.predict(w)[0], 3))
