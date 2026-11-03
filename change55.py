import pyarrow.fs as pafs, fn_root_cause as rc, pandas as pd
d = rc.dset(2026, pafs.S3FileSystem())
parts, n = [], 0
for b in d.to_batches(batch_size=250_000):
    t = rc.prep(b.to_pandas().loc[:, lambda x: ~x.columns.duplicated()][["CSN","PAT_ID","SERVICEDATE","FINAL_CPT","PAYOR","PLAN",
                 "FINAL_MODIFIERS","PLACE_OF_SERVICE_CODE","GROUP_LABEL","FILING_ORDER_C","LINE_AUTHORIZATION_NUMBER"]])
    parts.append(t[t.primary]); n += len(t)
    if n > 2_000_000: break
s = pd.concat(parts)
v = s.groupby("CSN").agg(n_cpt=("FINAL_CPT", "nunique"), n_auth_lines=("AUTH", lambda a: (a != "").sum()),
                         n_lines=("AUTH", "size"), n_auth_values=("AUTH", lambda a: a[a != ""].nunique()))
v = v[(v.n_cpt >= 2) & (v.n_auth_lines > 0)]
v["pattern"] = pd.cut(v.n_auth_lines / v.n_lines, [0, .999, 1.0], labels=["some lines blank", "all lines same/any auth"])
print("visits with 2+ CPTs and an auth:", len(v))
print(v.pattern.value_counts(normalize=True).round(3))
print("visits with 2+ different auth values:", (v.n_auth_values >= 2).mean().round(3))
pp = s[s.AUTH != ""].groupby("AUTH").PAT_ID.nunique().sort_values(ascending=False)
print("auth values used by >5 patients (placeholders?):", (pp > 5).sum()); print(pp.head(10))
