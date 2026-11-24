python - <<'EOF'
import pandas as pd
t = pd.read_parquet('exp_feat_core_s42/test_preds.parquet'); p = [c for c in t if c.startswith('p_')][0]
t['FINAL_CPT'] = t.FINAL_CPT.astype(str).str.strip().str.upper()
t = t[~t.FINAL_CPT.str.fullmatch(r'\d{4}F')]
s = t.groupby(['CSN','FINAL_CPT']).agg(y=('y','max'), s=(p,'max'),
        payer=('PAYOR_CLEAN','first'), plan=('PAYOR_PLAN_CLEAN','first')).reset_index()
code = pd.to_numeric(s.FINAL_CPT.where(s.FINAL_CPT.str.fullmatch(r'\d{5}')), errors='coerce')
em = s[code.between(99202, 99499)].copy()
cut = 0.5317
em['flag'] = em.s >= cut; em['FP'] = em.flag & (em.y == 0); em['TP'] = em.flag & (em.y == 1)
print(f"E/M services {len(em):,} | PA {int(em.y.sum()):,} ({em.y.mean():.2%}) | flagged {int(em.flag.sum()):,} | FP {int(em.FP.sum()):,} | TP {int(em.TP.sum()):,}")
for col in ['payer', 'plan', 'FINAL_CPT']:
    g = em.groupby(col).agg(services=('y','size'), pa=('y','sum'), FP=('FP','sum'), TP=('TP','sum'))
    g['pa_rate'] = (g.pa / g.services).round(4); g['share_of_EM_FP'] = (g.FP / em.FP.sum()).round(3)
    print(f"\n== E/M by {col} (top 15 by false alarms)"); print(g.sort_values('FP', ascending=False).head(15).to_string())
EOF
