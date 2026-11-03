import pyarrow.fs as pafs, fn_root_cause as rc
from cell_label_check import toks
d = rc.dset(2026, pafs.S3FileSystem())
t = d.to_table(filter=rc.isin(d, "CSN", u.CSN.unique())).to_pandas()
t = t.loc[:, ~t.columns.duplicated()]
t["CSN"] = t.CSN.astype(str).str.replace(r"\.0$", "", regex=True)
x = t.merge(u[["CSN", "FINAL_CPT", "t1_source"]], on=["CSN", "FINAL_CPT"])
x = x[x.GROUP_LABEL.astype(str).str.upper().isin(["TRUE", "1"])].drop_duplicates(["CSN", "FINAL_CPT"])
x["codes"] = [sorted(toks(a) | toks(b)) for a, b in zip(x.CARC.astype(object).fillna(""), x.get("RARC", "").astype(object).fillna(""))]
e = x.explode("codes")
print(pd.crosstab(e.codes, e.t1_source, margins=True).sort_values("All", ascending=False).head(20))
print(pd.crosstab(e.FINAL_CPT, e.codes).loc[lambda m: m.sum(1).sort_values(ascending=False).index[:12],
      lambda m: m.sum().sort_values(ascending=False).index[:10]])



python cell_label_check.py --cpt 99396 --plan "ANTHEM BCBS | ANTHEM BCBS COMMERCIAL" --out cell_check_anthem99396
python cell_label_check.py --cpt 99213 --plan "EMBLEM HEALTH/HIP | EMBLEM HEALTH/HIP" --out cell_check_emblem99213
