import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def metrics(y, p, thr):
    pr = p >= thr
    tp = int((pr & (y==1)).sum()); fp = int((pr & (y==0)).sum()); fn = int((~pr & (y==1)).sum()); tn = int((~pr & (y==0)).sum())
    prec = tp/(tp+fp) if tp+fp else 0; rec = tp/(tp+fn); f1 = 2*prec*rec/(prec+rec) if prec+rec else 0
    return dict(best_thr=thr, Accuracy=(tp+tn)/len(y), Precision=prec, Recall=rec, F1=f1,
                **{"AUC-ROC": roc_auc_score(y, p), "AUC-PR": average_precision_score(y, p)}, FP=fp, FN=fn, TP=tp, TN=tn)

rows = []
for run in ["exp_d1_s30", "exp_d1_dx", "exp_d1_dx3"]:
    t = pd.read_parquet(f"{run}/test_preds.parquet")
    for m in [c for c in t.columns if c.startswith("p_")]:
        if run != "exp_d1_s30" and "ctx" not in m:
            continue                                   # other models repeat the first run
        e = t.groupby("CSN").agg(y=("y","max"), p=(m,"max"))
        y, p = e.y.values, e.p.values
        best = max(np.round(np.arange(0.05, 0.96, 0.05), 2), key=lambda th: metrics(y, p, th)["F1"])
        rows.append(dict(run=run, model=m[2:], **metrics(y, p, best)))
out = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(out.round(4).to_string(index=False))
out.round(4).to_csv("experiment_log_metrics.csv", index=False)
