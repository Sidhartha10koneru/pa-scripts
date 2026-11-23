python - <<'EOF'
import lightgbm as lgb, pandas as pd
m = lgb.Booster(model_file='exp_feat_all_s42/models/lgb_ctx_full.txt')
g = pd.Series(m.feature_importance('gain'), index=m.feature_name())
print((g / g.sum()).sort_values(ascending=False).round(3).to_string())
EOF

python - <<'EOF'
import pandas as pd, pyarrow.fs as f
p = 'cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v8_draft/ai_prior_auth_09282026_v3_2025_std.parquet'
d = pd.read_parquet(p, columns=['VISITTYPE', 'SPECIALTY', 'LABEL_V6_PAX'], filesystem=f.S3FileSystem())
d['pa'] = d.LABEL_V6_PAX.astype(str).str.upper().eq('TRUE')
for c in ['VISITTYPE', 'SPECIALTY']:
    t = d.groupby(c).pa.agg(lines='size', pa_rate='mean').sort_values('lines', ascending=False)
    print(f'\n== {c}: {len(t):,} values'); print(t.head(25).round(3).to_string())
    print('-- values with >= 500 lines and PA rate >= 50%:'); print(t[(t.lines >= 500) & (t.pa_rate >= .5)].round(3).to_string())
EOF
