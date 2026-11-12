python - <<'EOF'
import numpy as np, pandas as pd
def svc(run):
    t = pd.read_parquet(f"{run}/test_preds.parquet"); p = [c for c in t.columns if c.startswith("p_")][0]
    t["CSN"] = t.CSN.astype(str).str.replace(r"\.0$","",regex=True); t["FINAL_CPT"] = t.FINAL_CPT.astype(str).str.strip().str.upper()
    t = t[~t.FINAL_CPT.str.fullmatch(r"\d{4}F")]
    return t.groupby(["CSN","FINAL_CPT"], observed=True).agg(y=("y","max"), s=(p,"max"))
fz = pd.read_parquet("frozen/test_v1_2026-05_2026-07/test_set_services.parquet"); fz["CSN"] = fz.CSN.astype(str).str.replace(r"\.0$","",regex=True)
keep = fz[fz.IS_QUALITY_CODE == 0].set_index(["CSN","FINAL_CPT"]).index
for seed, bench, pax in [("s42","exp_nofilter_2022","exp_v8d_l6pax_s42"), ("s7","exp_keep_s7","exp_v8d_l6pax_s7")]:
    d = svc(pax).join(svc(bench)[["s"]].rename(columns={"s":"sb"}), how="inner"); d = d[d.index.isin(keep)]
    y = d.y.to_numpy() == 1; P = int(y.sum())                                       # corrected labels
    for name, col in [("bench","sb"), ("new","s")]:
        d[f"r_{name}"] = d[col].rank(ascending=False, method="first")              # position in the flag list
    nb = int((d.sb >= np.quantile(d.sb[y], 0.2)).sum())                            # benchmark flags at 80% recall
    for name in ("bench","new"):
        tp = int(((d[f"r_{name}"] <= nb) & y).sum())
        print(f"{seed} same flags {nb:,} | {name:5s}: PA caught {tp:,} | missed {P-tp:,} | recall {tp/P:.1%}")
    for rec in (0.80, 0.85, 0.90):
        for name, col in [("bench","sb"), ("new","s")]:
            cut = np.quantile(d[col][y], 1-rec); f = d[col] >= cut
            print(f"{seed} recall {rec:.0%} | {name:5s}: flags {int(f.sum()):,} | false alarms {int((f & ~y).sum()):,} | precision {(f & y).sum()/f.sum():.1%}")
EOF
