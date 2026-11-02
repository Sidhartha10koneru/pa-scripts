import numpy as np, pandas as pd, pyarrow.dataset as ds, pyarrow.fs as pafs
from sklearn.metrics import roc_auc_score
import pa_base_adapter as pb

fs = pafs.S3FileSystem()
parts = []
for y in (2025, 2026):
    dset = ds.dataset(pb.BASE_PATH.format(year=y).replace("s3://", ""), filesystem=fs, format="parquet")
    parts.append(dset.to_table(filter=ds.field("FINAL_CPT") == "99214").to_pandas())   # every column, 99214 only
d = pd.concat(parts, ignore_index=True)
d = pb.clean_names(d)
d["L"] = d.GROUP_LABEL.astype(str).str.upper()

# the cell
x = d[(d.PAYOR_PLAN_CLEAN == "EMBLEM HEALTH/HIP | EMBLEM HEALTH/HIP") &
      (d.PLACE_OF_SERVICE_CODE.astype(str) == "11") &
      (d.FINAL_MODIFIERS.isna() | d.FINAL_MODIFIERS.astype(str).str.strip().isin(["", "nan", "None"])) &
      d.L.isin(["TRUE", "FALSE"])].copy()
x["y"] = (x.L == "TRUE").astype(int)
print("cell lines:", len(x), "| PA:", x.y.sum(), f"| rate {x.y.mean():.1%}")

# drop the label itself and anything that comes from the payer's response (would leak the answer)
leak = r"LABEL|AUTH|CARC|RARC|REMIT|DENI|PAID|PAYMENT|ALLOW|ADJUST|CLAIM_STATUS|CHECK|GROUP_GUID|CSN|^y$|^L$"
cand = [c for c in x.columns if not pd.Series([c]).str.contains(leak, case=False).iloc[0]]

# for each column: how well does "PA rate of its value" separate TRUE from FALSE? (2-fold, so no self-scoring)
rng = np.random.default_rng(0)
fold = rng.integers(0, 2, len(x))
rows = []
for c in cand:
    v = x[c].astype(str).fillna("NA")
    if v.nunique() < 2:
        continue
    p = np.zeros(len(x))
    for f in (0, 1):
        rate = x.y[fold != f].groupby(v[fold != f]).agg(["sum", "size"])
        sm = (rate["sum"] + 20 * x.y.mean()) / (rate["size"] + 20)                 # smoothed rate per value
        p[fold == f] = v[fold == f].map(sm).fillna(x.y.mean()).values
    rows.append({"column": c, "n_values": v.nunique(), "AUC": roc_auc_score(x.y, p),
                 "top_value_PA_rate": x.groupby(v).y.mean().max()})
r = pd.DataFrame(rows).sort_values("AUC", ascending=False)
pd.set_option("display.width", 200)
print(r.head(25).round(3).to_string(index=False))

# detail for the best few columns: PA rate per value
for c in r.column.head(5):
    t = x.groupby(x[c].astype(str)).y.agg(lines="size", pa="sum", pa_rate="mean").sort_values("pa", ascending=False)
    print(f"\n== {c}\n", t.head(10).round(3).to_string())
