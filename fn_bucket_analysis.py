# %% [markdown]
# # FN bucket analysis (notebook style: run cell by cell)
#
# **Question:** the model misses some PA services (false negatives). *Why* is each one missed?
#
# **Four data sources, labelled in every cell:**
# - **[TRAIN]**: the training data exactly as the model saw it: std v4 files, AMBIGUOUS dropped,
#   all filing orders, repeats removed (TRUE copy kept), service date before 2026-05-01.
# - **[TEST]**: `test_preds_with_guid.parquet`: every test line (May-Jul 2026) with its label `y`
#   and the model's score. The **FN are found here**.
# - **[V4]**: the std v4 files with *all* columns, before any prep. Used only to drill into one cell.
# - **[REMIT]**: the raw v3 remit files (several remit rows per line). Not used here; see `cell_label_check.py`.
#
# **Unit:** a *service* = CSN + CPT. A service is PA if any of its lines is TRUE; its score is the highest line score.
#
# **FN (false negative)** = a service that **is PA** (`y = 1`) but whose score is **below the cut-off**, so the model
# says "no PA". We use the cut-off that catches 80% of PA services (the operating point), not the best-F1 cut-off.

# %%
# Cell 1 - settings
import re
import numpy as np
import pandas as pd
import pyarrow.fs as pafs
import pa_base_adapter as pb
import pa_experiments as pe

TEST_FILE = "exp_nofilter_2022/test_preds_with_guid.parquet"   # [TEST] lines + scores
MODEL_COL = "p_lgb_full_2022"            # score column: 2022-2026 model ("p_lgb_full_2025" = benchmark)
TRAIN_YEARS = [2022, 2023, 2024, 2025, 2026]   # must match the model's training years (2025 model: [2025, 2026])
TEST_START = "2026-05-01"
RECALL = 0.80                            # operating point: catch 80% of PA services
MIN_N = 20                               # "rare cell" threshold (the model's own leaf minimum is 100)
KEY = ["FINAL_CPT", "FINAL_MODIFIERS", "PLACE_OF_SERVICE_CODE", "PAYOR_PLAN_CLEAN"]   # the model's cell (5 features)
OUT = "exp_nofilter_2022"
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)

# %% [markdown]
# ## Cell 2 - [TRAIN] load the training data
# Same preparation as the model run: no filing-order filter, AMBIGUOUS dropped, repeats removed keeping the TRUE copy.
# Then keep only lines before the test window. This is what the model learned from.
# Takes a few minutes and a lot of memory for 5 years.

# %%
pb.INCLUDE_SECONDARY, pb.DEDUP_KEEP_TRUE = True, True
cols = [pb.CSN, pb.LABEL, pb.DATE, pb.FILING, "FINAL_CPT", "FINAL_MODIFIERS", "PLACE_OF_SERVICE_CODE", "PAYOR", "PLAN"]
d = pe.add_dates(pb.load(TRAIN_YEARS, cols, 1.0, pafs.S3FileSystem()))
d = d.loc[:, ~d.columns.duplicated()]
tr = d[d.SVC_DATE < TEST_START].copy()
for c in KEY:
    tr[c] = tr[c].astype(object).fillna("").astype(str)
print(f"[TRAIN] {len(tr):,} lines | PA rate {tr.y.mean():.2%} | {tr.SVC_DATE.min():%Y-%m-%d} .. {tr.SVC_DATE.max():%Y-%m-%d}")

# %% [markdown]
# ## Cell 3 - [TRAIN] history of every cell
# For each 5-feature cell: how many training lines (`train_n`) and what share were PA (`train_pa_rate`).
# Because all lines of a cell have identical features, the model's score for a cell is close to this rate.

# %%
cell = tr.groupby(KEY).y.agg(train_n="size", train_pa_rate="mean")
cpt_rate = tr.groupby("FINAL_CPT").y.mean().rename("train_cpt_pa_rate")
print(f"[TRAIN] {len(cell):,} distinct cells | cells with < {MIN_N} lines: {(cell.train_n < MIN_N).mean():.1%}")

# %% [markdown]
# ## Cell 4 - [TEST] load the test lines and build services
# Every test line has its label `y` (from GROUP_LABEL) and the model's score.
# A service = CSN + CPT. Label = max of its lines (PA if any line is TRUE). Score = max of its lines.
# The modifier / POS / plan of the service = those of its highest-scoring line (needed to look up its cell).

# %%
t = pd.read_parquet(TEST_FILE)
t = t.loc[:, ~t.columns.duplicated()]
for c in KEY:
    t[c] = t[c].astype(object).fillna("").astype(str)
s = t.groupby(["CSN", "FINAL_CPT"]).agg(y=("y", "max"), score=(MODEL_COL, "max"), n_lines=("y", "size")).reset_index()
top = t.loc[t.groupby(["CSN", "FINAL_CPT"])[MODEL_COL].idxmax(), ["CSN"] + KEY]
s = s.merge(top, on=["CSN", "FINAL_CPT"])
print(f"[TEST] {len(t):,} lines -> {len(s):,} services | PA services {int(s.y.sum()):,}")

# %% [markdown]
# ## Cell 5 - [TEST] the cut-off and the FN
# Cut-off = the score at which 80% of PA services are caught (computed on the test services).
# **FN = PA service (y = 1) with score below the cut-off.** The confusion counts are printed as a check
# (they should match the experiment sheet's 80%-recall row for this model).

# %%
cut = float(np.quantile(s.score[s.y == 1], 1 - RECALL))
flag = s.score >= cut
tp, fp = int((flag & (s.y == 1)).sum()), int((flag & (s.y == 0)).sum())
fn_n, tn = int((~flag & (s.y == 1)).sum()), int((~flag & (s.y == 0)).sum())
print(f"cut-off {cut:.4f} | TP {tp:,}  FP {fp:,}  FN {fn_n:,}  TN {tn:,} | recall {tp/(tp+fn_n):.3f} precision {tp/(tp+fp):.3f}")
fn = s[(s.y == 1) & (s.score < cut)].copy()          # <- the FN services we explain below
print(f"[TEST] FN services: {len(fn):,}")

# %% [markdown]
# ## Cell 6 - [TEST FN] + [TRAIN] attach each FN's training history
# For every FN service: how often its exact cell was seen in training, and how often it was PA there.
# Cells never seen in training get train_n = 0.

# %%
fn = fn.join(cell, on=KEY).join(cpt_rate, on="FINAL_CPT")
fn["train_n"] = fn.train_n.fillna(0)
print(fn[["train_n", "train_pa_rate", "score"]].describe().round(3))

# %% [markdown]
# ## Cell 7 - [TEST] "passenger" signal
# CPT family by code range (quality F codes, E/M visits, labs, "anchor" = imaging / procedure / J-drug / PT).
# `anchor_in_visit` = the same **test** visit (CSN) also has a PA service from an anchor family.
# This is a heuristic: it does NOT check the auth numbers (that needs [REMIT], see cell_label_check.py).

# %%
def fam(c):
    c = str(c).upper()
    if re.fullmatch(r"\d{4}F", c): return "quality"
    if re.fullmatch(r"99[2-4]\d\d|G2211", c): return "E/M"
    if re.fullmatch(r"8\d{4}|36415", c): return "lab"
    if re.fullmatch(r"7\d{4}|[1-6]\d{4}|J\d{4}|97\d{3}", c): return "anchor"
    return "other"

pa_svc = s[s.y == 1].assign(f=lambda x: x.FINAL_CPT.map(fam))
anchor_visits = set(pa_svc.loc[pa_svc.f == "anchor", "CSN"])
fn["family"] = fn.FINAL_CPT.map(fam)
fn["anchor_in_visit"] = fn.CSN.isin(anchor_visits) & (fn.family != "anchor")
print(pd.crosstab(fn.family, fn.anchor_in_visit, margins=True))

# %% [markdown]
# ## Cell 8 - [TEST FN] assign buckets (first rule that matches wins)
# 1. **Likely label noise**: quality code, or E/M / lab on a visit with a PA anchor service (heuristic, cell 7)
# 2. **Unseen or rare cell**: fewer than MIN_N training lines for this exact cell
# 3. **Historically non-PA**: enough history, training PA rate < 5%
# 4. **Mixed cell**: training PA rate 5-50% (same features, both outcomes common)
# 5. **Model miss**: training PA rate >= 50%, yet scored below the cut-off

# %%
fn["bucket"] = np.select(
    [fn.family.eq("quality") | (fn.family.isin(["E/M", "lab"]) & fn.anchor_in_visit),
     fn.train_n < MIN_N, fn.train_pa_rate < 0.05, fn.train_pa_rate < 0.50],
    ["1 likely label noise", "2 unseen or rare cell", "3 historically non-PA", "4 mixed cell"],
    "5 model miss")
summary = fn.groupby("bucket").agg(services=("CSN", "size"), median_score=("score", "median"),
                                   median_train_rate=("train_pa_rate", "median"), median_train_n=("train_n", "median"))
summary["share"] = (summary.services / summary.services.sum()).round(3)
print(summary.round(3).to_string())
for b, g in fn.groupby("bucket"):
    print(f"\n{b}: top CPTs", g.FINAL_CPT.value_counts().head(8).to_dict())
fn.to_csv(f"{OUT}/fn_buckets_recall{int(RECALL*100)}_min{MIN_N}.csv", index=False)

# %% [markdown]
# ## Cell 9 - [TEST] + [TRAIN] the mixed cells (bucket 4)
# Which cells hold the mixed FN, their training rate vs their rate in the **whole test set** (not only FN),
# and how concentrated bucket 4 is. Training rate far from test rate = drift or label inflation.

# %%
m = fn[fn.bucket.str.startswith("4")]
cells = (m.groupby(KEY).agg(fn_services=("CSN", "size"), train_n=("train_n", "first"),
                            train_pa_rate=("train_pa_rate", "first"), median_score=("score", "median"))
          .sort_values("fn_services", ascending=False))
test_cell = s.groupby(KEY).agg(test_services=("y", "size"), test_pa=("y", "sum"))
cells = cells.join(test_cell)
cells["test_pa_rate"] = cells.test_pa / cells.test_services
cells["cum_share_of_bucket4"] = cells.fn_services.cumsum() / cells.fn_services.sum()
print(len(cells), "mixed cells behind", len(m), "FN services")
print(cells.head(30).round(3).to_string())
cells.to_csv(f"{OUT}/fn_mixed_cells_recall{int(RECALL*100)}.csv")

# %% [markdown]
# ## Cell 10 - [V4] drill into ONE cell with all columns (2025-2026)
# Loads every column of the std v4 files for one CPT, keeps one cell, and ranks every other column by how well
# it separates PA from non-PA **inside** the cell (2-fold, so a column can't score by memorising its own rows).
# Columns built from the label (N_TRUE, N_FALSE, N_AMBIGUOUS, LEVEL*_LABEL, auth / remit fields) are excluded:
# they would "predict" the label because they are part of it.

# %%
import pyarrow.dataset as ds
from sklearn.metrics import roc_auc_score

CELL_CPT, CELL_PLAN, CELL_POS, CELL_MOD = "99214", "EMBLEM HEALTH/HIP | EMBLEM HEALTH/HIP", "11", ""
fs = pafs.S3FileSystem()
v = pd.concat([ds.dataset(pb.BASE_PATH.format(year=yy).replace("s3://", ""), filesystem=fs, format="parquet")
               .to_table(filter=ds.field("FINAL_CPT") == CELL_CPT).to_pandas() for yy in (2025, 2026)], ignore_index=True)
v = pb.clean_names(v.loc[:, ~v.columns.duplicated()])
mod = v.FINAL_MODIFIERS.astype(object).fillna("").astype(str).str.strip()
x = v[(v.PAYOR_PLAN_CLEAN == CELL_PLAN) & (v.PLACE_OF_SERVICE_CODE.astype(str) == CELL_POS) & (mod == CELL_MOD)
      & v.GROUP_LABEL.astype(str).str.upper().isin(["TRUE", "FALSE"])].copy()
x["y"] = (x.GROUP_LABEL.astype(str).str.upper() == "TRUE").astype(int)
print(f"[V4] cell lines {len(x):,} | PA {x.y.sum():,} | rate {x.y.mean():.2%}")

leak = r"LABEL|^N_(TRUE|FALSE|AMBIGUOUS)$|AUTH|CARC|RARC|REMIT|DENI|PAID|PAYMENT|ALLOW|ADJUST|CLAIM_SUBMISSION|GROUP_GUID|GROUP_SIZE|^CSN|^y$"
cand = [c for c in x.columns if not re.search(leak, c, re.I)]
fold = np.random.default_rng(0).integers(0, 2, len(x))
rows = []
for c in cand:
    val = x[c].astype(object).fillna("NA").astype(str)
    if val.nunique() < 2:
        continue
    p = np.zeros(len(x))
    for f in (0, 1):
        r = x.y[fold != f].groupby(val[fold != f]).agg(["sum", "size"])
        sm = (r["sum"] + 20 * x.y.mean()) / (r["size"] + 20)
        p[fold == f] = val[fold == f].map(sm).fillna(x.y.mean()).values
    rows.append({"column": c, "n_values": val.nunique(), "AUC": roc_auc_score(x.y, p)})
print(pd.DataFrame(rows).sort_values("AUC", ascending=False).head(25).round(3).to_string(index=False))

# %% [markdown]
# ## Cell 11 - [V4] same patient over time, and how each label was made
# `prior_pa` = the patient already had a PA visit **in this cell** before this one (earlier dates only).
# Then: how the FALSE lines were decided (LEVEL1/LEVEL2 = first remit row in the std file) and the auth field.
# For all remit rows per line, run cell_label_check.py ([REMIT]).

# %%
x["SVC"] = pd.to_datetime(x.SERVICEDATE, errors="coerce")
x = x.sort_values(["PAT_ID", "SVC"])
x["prior_pa"] = x.groupby("PAT_ID").y.transform(lambda q: q.shift().cummax()).fillna(0)
print(x.groupby("prior_pa").y.agg(lines="size", pa="sum", pa_rate="mean"))
x["group"] = np.select([x.groupby("PAT_ID").y.transform("max") == 0, x.prior_pa == 1],
                       ["patient never PA", "after a PA visit"], "PA patient, before/at PA")
f = x[x.y == 0]
print(pd.crosstab([f.LEVEL1_LABEL.astype(str), f.LEVEL2_LABEL.astype(str)], f.group, margins=True))
print(f.LINE_AUTHORIZATION_NUMBER.astype(object).fillna("(blank)").astype(str).replace("", "(blank)").value_counts().head(10))
