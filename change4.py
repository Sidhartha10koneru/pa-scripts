python - <<'EOF'
import numpy as np, pandas as pd
fz = pd.read_parquet("frozen/test_v1_2026-05_2026-07/test_set_services.parquet")
fz = fz[fz.IS_QUALITY_CODE == 0]; fz["CSN"] = fz.CSN.astype(str)
for run in ["exp_qc_exclude_rv", "exp_v8d_l7spax_s42"]:
    t = pd.read_parquet(f"{run}/test_preds.parquet"); p = [c for c in t.columns if c.startswith("p_")][0]
    t["CSN"] = t.CSN.astype(str); t["FINAL_CPT"] = t.FINAL_CPT.astype(str).str.strip().str.upper()
    s = t.groupby(["CSN", "FINAL_CPT"], observed=True)[p].max().rename("s").reset_index()
    m = fz[["CSN", "FINAL_CPT", "y"]].merge(s, on=["CSN", "FINAL_CPT"])
    for lvl, d in [("service", m), ("visit", m.groupby("CSN").agg(y=("y", "max"), s=("s", "max")))]:
        y, sc = d.y.to_numpy(), d.s.to_numpy()
        for rec in (0.75, 0.80):
            f = sc >= np.quantile(sc[y == 1], 1 - rec); tp = (f & (y == 1)).sum(); fp = (f & (y == 0)).sum()
            print(f"{run:22s} {lvl:7s} recall {rec:.0%}: precision {tp/(tp+fp):.1%}  FP {fp:,}  flagged {f.sum():,} of {len(y):,}")
EOF
