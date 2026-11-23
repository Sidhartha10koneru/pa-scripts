python - <<'EOF'
import pandas as pd, numpy as np
t = pd.read_parquet('exp_feat_core_s42/test_preds.parquet'); p = [c for c in t if c.startswith('p_')][0]
t = t[~t.FINAL_CPT.astype(str).str.fullmatch(r'\d{4}F')]
s = t.groupby(['CSN','FINAL_CPT']).agg(y=('y','max'), s=(p,'max')).reset_index()
cut = np.quantile(s.s[s.y==1], 0.15); s['flag'] = s.s >= cut
fn = s[(s.y==1) & ~s.flag]
vf = s.groupby('CSN').flag.any()
print(f"missed PA services {len(fn):,}; in a visit where another service IS flagged: {fn.CSN.map(vf).mean():.1%}")
v = s.groupby('CSN').agg(y=('y','max'), f=('flag','any'))
print(f"visit level: PA visits {int(v.y.sum()):,}, caught {v.f[v.y==1].mean():.1%}, missed {int((v.y==1).sum() - (v.f & (v.y==1)).sum()):,}")
EOF
