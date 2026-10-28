import glob, json, os, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
from pa_base_adapter import apply_categories

feats = ["FINAL_CPT","FINAL_MODIFIERS","PLACE_OF_SERVICE_CODE","PAYOR_CLEAN","PAYOR_PLAN_CLEAN","RULE_PLAN_SHARE"]
KEYS  = {"cpt_plan": ["FINAL_CPT","PAYOR_PLAN_CLEAN"],
         "prd_cell": ["FINAL_CPT","FINAL_MODIFIERS","PLACE_OF_SERVICE_CODE","PAYOR_PLAN_CLEAN"]}
sig = lambda z: 1 / (1 + np.exp(-z))

t    = pd.read_parquet("exp_d1_s30/test_preds.parquet")                     # May–Jul test lines
base = lgb.Booster(model_file="exp_d2_cells/base_frozen.txt")
cats = json.load(open("exp_d2_cells/base_frozen_categories.json"))
raw  = base.predict(apply_categories(t, cats, feats), raw_score=True)

def metrics(y, p, thr):
    pr = p >= thr
    tp = int((pr & (y==1)).sum()); fp = int((pr & (y==0)).sum()); fn = int((~pr & (y==1)).sum()); tn = int((~pr & (y==0)).sum())
    prec = tp/(tp+fp) if tp+fp else 0; rec = tp/(tp+fn); f1 = 2*prec*rec/(prec+rec) if prec+rec else 0
    return dict(best_thr=thr, Accuracy=(tp+tn)/len(y), Precision=prec, Recall=rec, F1=f1, FP=fp, FN=fn, TP=tp, TN=tn)

def visit_row(name, p, extra=""):
    e = pd.DataFrame({"y": t.y.values, "p": p, "c": t.CSN.values}).groupby("c").max()
    y, s = e.y.values, e.p.values
    best = max(np.round(np.arange(0.05, 0.96, 0.05), 2), key=lambda th: metrics(y, s, th)["F1"])
    return dict(model=name, notes=extra, **{"AUC-ROC": roc_auc_score(y, s), "AUC-PR": average_precision_score(y, s)},
                **metrics(y, s, best))

rows = [visit_row("frozen (to Jan 2026, no update)", sig(raw)),
        visit_row("full retrain (to Apr 2026)", t.p_lgb_rules_full.values)]
for f in sorted(glob.glob("exp_d2_cells/weights_*_k*.csv")):
    kname = "prd_cell" if "prd_cell" in f else "cpt_plan"
    k = os.path.basename(f).split("_k")[1].replace(".csv", "")
    w = pd.read_csv(f, index_col="cell")["weight"]
    cell = t[KEYS[kname]].astype(str).agg("|".join, axis=1)
    matched = cell.isin(w.index).mean()
    rows.append(visit_row(f"cell weights {kname} k={k}", sig(raw + cell.map(w).fillna(0).values),
                          f"test lines with a weight: {matched:.1%}"))

out = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(out.round(4).to_string(index=False))
out.round(4).to_csv("experiment_log_cell_weights.csv", index=False)
