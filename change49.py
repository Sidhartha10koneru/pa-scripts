import json, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import average_precision_score
import pa_base_adapter as pb

t = pd.read_parquet("exp_nofilter_2025/test_preds.parquet")            # test WITH secondary claims
m = lgb.Booster(model_file="exp_base_2025_2026/models/lgb_full.txt")    # benchmark model
cats = json.load(open("exp_base_2025_2026/models/lgb_full_categories.json"))
t["p_benchmark"] = m.predict(pb.apply_categories(t, cats, pb.BASE_FEATS))
t = t.rename(columns={"p_lgb_full": "p_nofilter"})

def at(y, s, cut):
    f = s >= cut; tp = int((f & (y == 1)).sum()); fp = int((f & (y == 0)).sum())
    fn = int((~f & (y == 1)).sum()); tn = len(y) - tp - fp - fn
    return dict(cutoff=round(float(cut), 4), recall=tp / (tp + fn), precision=tp / max(tp + fp, 1), TP=tp, FP=fp, FN=fn, TN=tn)

rows = []
for lv, keys in {"service": [pb.CSN, pb.CPT], "visit": [pb.CSN]}.items():
    g = t.groupby(keys).agg(y=("y", "max"), b=("p_benchmark", "max"), n=("p_nofilter", "max"))
    for model in ("b", "n"):
        y, s = g.y.values, g[model].values
        name = "Benchmark model" if model == "b" else "No-filter model"
        rows.append({"level": lv, "model": name, "point": "AUC-PR", "AUC-PR": average_precision_score(y, s)})
        for r in (0.8, 0.9):
            rows.append({"level": lv, "model": name, "point": f"{r:.0%} recall", **at(y, s, np.quantile(s[y == 1], 1 - r))})
pd.set_option("display.width", 200)
print(pd.DataFrame(rows).round(4).to_string(index=False))
