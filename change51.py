import numpy as np, pyarrow.fs as pafs, fn_root_cause as rc, pandas as pd
d = rc.dset(2026, pafs.S3FileSystem())
cols = ["CSN","PAT_ID","SERVICEDATE","FINAL_CPT","GROUP_LABEL","LEVEL1_LABEL","LEVEL2_LABEL","CARC","RARC","LINE_AUTHORIZATION_NUMBER"]
parts, n = [], 0
for b in d.to_batches(batch_size=250_000):
    t = b.to_pandas(); t = t.loc[:, ~t.columns.duplicated()]; parts.append(t[[c for c in cols if c in t]]); n += len(t)
    if n > 2_000_000: break
s = pd.concat(parts, ignore_index=True)
up = lambda c: s[c].astype(object).fillna("").astype(str).str.upper().str.strip()
codes, au, L1, L = up("CARC") + " " + up("RARC"), up("LINE_AUTHORIZATION_NUMBER").str.rstrip(",. "), up("LEVEL1_LABEL"), up("GROUP_LABEL")

# 1) N54 / M62: exact code vs substring only
exact = codes.str.contains(r"(?<![A-Z0-9])(?:N54|M62)(?!\d)")
sub = codes.str.contains("N54|M62") & ~exact
print("1) lines TRUE via N54/M62 substring only:", int((sub & up("LEVEL2_LABEL").eq("TRUE")).sum()))
print(codes[sub].value_counts().head(10))

# 2) auth value that is a date (MMDDYYYY), and whether it equals the service date
isdate = au.str.fullmatch(r"(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])20[12]\d")
same = isdate & (au == pd.to_datetime(s.SERVICEDATE, errors="coerce").dt.strftime("%m%d%Y"))
print("2) date-like auth values:\n", pd.crosstab([isdate.rename("is_date"), same.rename("= service date")], L1, margins=True))

# 3) one 'auth number' used by many patients but labelled TRUE
g = s[L1.eq("TRUE")].assign(au=au).groupby("au").PAT_ID.nunique().sort_values(ascending=False)
print("3) TRUE auth values used by >5 patients:", int((g > 5).sum())); print(g.head(15))
