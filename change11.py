python - <<'EOF'
import pandas as pd
from sklearn.metrics import average_precision_score as ap
def svc(run):
    t = pd.read_parquet(f"{run}/test_preds.parquet"); t = t.loc[:, ~t.columns.duplicated()]
    p = [c for c in t.columns if c.startswith("p_")][0]
    t["CSN"] = t.CSN.astype(str).str.replace(r"\.0$", "", regex=True)
    t = t[~t.FINAL_CPT.astype(str).str.fullmatch(r"\d{4}F")]
    return t.groupby(["CSN", "FINAL_CPT"]).agg(y=("y", "max"), s=(p, "max"))
keys = {k: svc(r) for k, r in {"base": "exp_v8d_grp_s42", "l6pax": "exp_v8d_l6pax_s42", "l6paxc": "exp_v9d_l6paxc_s42"}.items()}
idx = keys["base"].index.intersection(keys["l6pax"].index).intersection(keys["l6paxc"].index)
agree = idx[(keys["base"].y.loc[idx] == keys["l6pax"].y.loc[idx]) & (keys["l6pax"].y.loc[idx] == keys["l6paxc"].y.loc[idx])]
runs = {"Baseline (5 fields)": "exp_v8d_l6pax_s42", "Core s42": "exp_feat_core_s42", "Core s7": "exp_feat_core_s7",
        "Core + history s42": "exp_feat_hist_s42", "Core + history s7": "exp_feat_hist_s7", "Core + history, 30-day delay": "exp_feat_hist_lag30_s42"}
for name, r in runs.items():
    S = svc(r).reindex(agree).dropna()
    print(f"{name:30s} AUC-PR {ap(S.y, S.s):.4f}   (services {len(S):,}, PA {int(S.y.sum()):,})")
EOF
