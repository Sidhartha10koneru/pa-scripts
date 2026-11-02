import numpy as np, pandas as pd

p = pd.read_parquet("exp_nofilter_2025_ids/test_preds.parquet")
model = "p_lgb_full"                                   # score column

# 1) service level: one row per CSN + CPT (PA if any line is TRUE, score = highest line score)
s = p.groupby(["CSN", "FINAL_CPT"]).agg(y=("y", "max"), score=(model, "max"))
# visit level instead: s = p.groupby("CSN").agg(y=("y", "max"), score=(model, "max"))
y, sc = s.y.values, s.score.values

# 2) every cut-off 0.05 ... 0.95
rows = []
for cut in np.round(np.arange(0.05, 0.96, 0.05), 2):
    flag = sc >= cut
    tp = int((flag & (y == 1)).sum()); fp = int((flag & (y == 0)).sum())
    fn = int((~flag & (y == 1)).sum()); tn = int((~flag & (y == 0)).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    rows.append({"cutoff": cut, "flagged": tp + fp, "precision": prec, "recall": rec, "F1": f1,
                 "TP": tp, "FP": fp, "FN": fn, "TN": tn, "accuracy": (tp + tn) / len(y)})
t = pd.DataFrame(rows)

# 3) best_thr = cut-off with the highest F1
best = t.loc[t.F1.idxmax()]
pd.set_option("display.width", 200)
print(t.round(4).to_string(index=False))
print(f"\nbest_thr = {best.cutoff}  |  precision {best.precision:.4f}  recall {best.recall:.4f}  F1 {best.F1:.4f}  "
      f"TP {best.TP:,}  FP {best.FP:,}  FN {best.FN:,}  TN {best.TN:,}")
