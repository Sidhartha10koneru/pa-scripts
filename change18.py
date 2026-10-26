import pandas as pd, numpy as np
t = pd.read_parquet("exp_d1_s30/test_preds.parquet")
e = t.groupby("CSN").agg(y=("y","max"), p=("p_lgb_rules_full","max")).sort_values("p", ascending=False)
e["caught"] = e.y.cumsum() / e.y.sum()                    # share of all PA visits flagged so far
e["reviewed"] = np.arange(1, len(e) + 1) / len(e)          # share of all visits flagged so far
e["precision"] = e.y.cumsum() / np.arange(1, len(e) + 1)
for target in [0.70, 0.80, 0.90, 0.95, 0.98, 0.99]:
    r = e[e.caught >= target].iloc[0]
    print(f"catch {target:.0%} of PA -> review {r.reviewed:.1%} of visits "
          f"({int(r.reviewed*len(e)):,} visits, precision {r.precision:.1%}, cut-off {r.p:.4f})")
