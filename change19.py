import pandas as pd, numpy as np
t = pd.read_parquet("exp_d1_s30/test_preds.parquet")
e = t.groupby("CSN").agg(y=("y","max"), p=("p_lgb_rules_full","max"))      # one row per visit
for th in [0.3, 0.5, 0.7, 0.9]:
    pr = e.p >= th
    tp = (pr & (e.y==1)).sum(); fp = (pr & (e.y==0)).sum(); fn = (~pr & (e.y==1)).sum(); tn = (~pr & (e.y==0)).sum()
    prec = tp/(tp+fp) if tp+fp else np.nan; rec = tp/(tp+fn)
    print(f"threshold {th}: accuracy {(tp+tn)/len(e):.3f} | precision {prec:.3f} | recall {rec:.3f} | "
          f"F1 {2*prec*rec/(prec+rec):.3f} | TP {tp} FP {fp} FN {fn} TN {tn}")
