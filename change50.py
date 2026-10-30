import numpy as np, pandas as pd
p = pd.read_parquet("exp_nofilter_2025/test_preds.parquet")
p["month"] = pd.to_datetime(p.SVC_DATE).dt.to_period("M").astype(str)
s = p.groupby(["CSN", "FINAL_CPT"]).agg(y=("y", "max"), sc=("p_lgb_full", "max"), month=("month", "min"))

def counts(y, f):
    tp = int((f & (y == 1)).sum()); fp = int((f & (y == 0)).sum()); fn = int((~f & (y == 1)).sum())
    return tp, fp, fn, tp / (tp + fn), tp / max(tp + fp, 1)

may = s[s.month == "2026-05"]
cut80 = np.quantile(may.sc[may.y == 1], 0.2)          # cut-off giving 80% recall, chosen on MAY only
rows = []
for m, g in s.groupby("month"):
    y, sc = g.y.values, g.sc.values
    tp, fp, fn, rec, prec = counts(y, sc >= cut80)                         # fixed score cut-off from May
    k = int(0.05 * len(g)); top = np.zeros(len(g), bool); top[np.argsort(-sc)[:k]] = True
    tp5, fp5, fn5, rec5, prec5 = counts(y, top)                              # top 5% of services, no threshold
    rows.append({"month": m, "services": len(g), "PA": int(y.sum()),
                 "cut_needed_for_80%": round(float(np.quantile(sc[y == 1], 0.2)), 3),
                 "fixed_cut_recall": round(rec, 3), "fixed_cut_precision": round(prec, 3), "fixed_cut_FP": fp,
                 "top5%_recall": round(rec5, 3), "top5%_precision": round(prec5, 3), "top5%_TP": tp5, "top5%_FP": fp5})
print(f"cut-off chosen on May for 80% recall: {cut80:.3f}\n", pd.DataFrame(rows).to_string(index=False))
