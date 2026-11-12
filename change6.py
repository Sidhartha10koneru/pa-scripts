python - <<'EOF'
import numpy as np, pandas as pd
def svc(run):
    t = pd.read_parquet(f"{run}/test_preds.parquet"); p = [c for c in t.columns if c.startswith("p_")][0]
    t["CSN"] = t.CSN.astype(str).str.replace(r"\.0$", "", regex=True); t["FINAL_CPT"] = t.FINAL_CPT.astype(str).str.strip().str.upper()
    t = t[~t.FINAL_CPT.str.fullmatch(r"\d{4}F")]
    return t.groupby(["CSN","FINAL_CPT"], observed=True).agg(y=("y","max"), s=(p,"max"))
fz = pd.read_parquet("frozen/test_v1_2026-05_2026-07/test_set_services.parquet"); fz["CSN"] = fz.CSN.astype(str).str.replace(r"\.0$","",regex=True)
keep = fz[fz.IS_QUALITY_CODE == 0].set_index(["CSN","FINAL_CPT"]).index
for seed, bench, pax in [("s42","exp_nofilter_2022","exp_v8d_l6pax_s42"), ("s7","exp_keep_s7","exp_v8d_l6pax_s7")]:
    P = svc(pax); B = svc(bench)[["s"]].rename(columns={"s":"sb"})
    d = P.join(B, how="inner"); d = d[d.index.isin(keep)]; y = d.y.to_numpy() == 1     # corrected labels
    sb, sp = d.sb.to_numpy(), d.s.to_numpy()
    budget = int(((sb >= np.quantile(sb[y], 0.2)) & ~y).sum())                        # benchmark FP at 80% recall
    neg = np.sort(sp[~y])[::-1]; cut = neg[budget - 1]                                   # new model's cut-off at the same FP
    fn_b, fn_p = int((sb[y] < np.quantile(sb[y], 0.2)).sum()), int((sp[y] < cut).sum())
    print(f"{seed}: FP budget {budget:,} | FN benchmark {fn_b:,} -> passenger model {fn_p:,} ({fn_p - fn_b:+,}) | recall {1-fn_b/y.sum():.1%} -> {1-fn_p/y.sum():.1%}")
EOF
