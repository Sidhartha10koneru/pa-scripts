python - <<'EOF'
import pandas as pd, pyarrow.fs as f
p = 'cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v9_draft/ai_prior_auth_09282026_v3_2026_std.parquet'
d = pd.read_parquet(p, columns=['FINAL_CPT','SERVICEDATE','LABEL_V6_PAX','LINE_AUTHORIZATION_NUMBER','CARC','RARC','LEVEL1_LABEL'],
                    filesystem=f.S3FileSystem())
dt = pd.to_datetime(d.SERVICEDATE, errors='coerce')
d = d[(dt >= '2026-05-01') & (dt < '2026-08-01') & ~d.FINAL_CPT.astype(str).str.fullmatch(r'\d{4}F')]
pa = d[d.LABEL_V6_PAX.astype(str).str.upper().eq('TRUE')]
txt = ~pa.LINE_AUTHORIZATION_NUMBER.astype(str).str.strip().str.upper().isin(['', 'NAN', 'NONE', 'NULL'])
print(f"test-window PA lines {len(pa):,} | with auth text {txt.mean():.1%} | without auth text {(~txt).sum():,}")
for c in ['CARC', 'RARC', 'LEVEL1_LABEL']:
    print(f"\n-- {c} on PA lines WITHOUT auth text"); print(pa[~txt][c].astype(str).value_counts().head(12).to_string())
print("\n-- LEVEL1_LABEL on all PA lines"); print(pa.LEVEL1_LABEL.astype(str).value_counts().head(10).to_string())
EOF


python - <<'EOF'
import pandas as pd, json
t = pd.read_csv('full_review_core/4_fn_top_PAYOR_CLEAN.csv')
print(t.head(12).to_string(index=False))
odd = t[(t.pa_caught == 0) & (t.FN > 50)].PAYOR_CLEAN.tolist()
cats = json.load(open('exp_feat_core_s42/models/lgb_ctx_full_categories.json'))
p = pd.read_parquet('exp_feat_core_s42/test_preds.parquet', columns=['PAYOR_CLEAN','PAYOR_PLAN_CLEAN','y'])
for name in odd:
    print(f"\n== {name} | payer seen in training: {name in set(cats['PAYOR_CLEAN'])}")
    s = p[p.PAYOR_CLEAN == name].groupby('PAYOR_PLAN_CLEAN').y.agg(lines='size', pa_rate='mean')
    s['plan_seen_in_training'] = s.index.isin(set(cats['PAYOR_PLAN_CLEAN']))
    print(s.sort_values('lines', ascending=False).head(10).round(3).to_string())
EOF
