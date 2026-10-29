import glob, json, os, numpy as np, pandas as pd, lightgbm as lgb, pyarrow.fs as pafs
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupShuffleSplit
import pa_base_adapter as pb, pa_experiments as pe
from pa_base_adapter import apply_categories

D1, D2, SAMPLE, TAG = "exp_d1_full", "exp_d2_cells_full", 1.0, "_full"
feats = ["FINAL_CPT","FINAL_MODIFIERS","PLACE_OF_SERVICE_CODE","PAYOR_CLEAN","PAYOR_PLAN_CLEAN","RULE_PLAN_SHARE"]
KEYS  = {"cpt_plan": ["FINAL_CPT","PAYOR_PLAN_CLEAN"], "prd_cell": ["FINAL_CPT","FINAL_MODIFIERS","PLACE_OF_SERVICE_CODE","PAYOR_PLAN_CLEAN"]}
sig = lambda z: 1 / (1 + np.exp(-z))

# ---------- metrics helpers (visit level, best-F1 threshold)
def metrics(y, p, thr):
    pr = p >= thr
    tp = int((pr & (y==1)).sum()); fp = int((pr & (y==0)).sum()); fn = int((~pr & (y==1)).sum()); tn = int((~pr & (y==0)).sum())
    prec = tp/(tp+fp) if tp+fp else 0; rec = tp/(tp+fn); f1 = 2*prec*rec/(prec+rec) if prec+rec else 0
    return dict(best_thr=thr, Accuracy=(tp+tn)/len(y), Precision=prec, Recall=rec, F1=f1, FP=fp, FN=fn, TP=tp, TN=tn)

def visit_row(name, p, csn, y_line, note=""):
    e = pd.DataFrame({"y": y_line, "p": p, "c": csn}).groupby("c").max()
    y, s = e.y.values, e.p.values
    best = max(np.round(np.arange(0.05, 0.96, 0.05), 2), key=lambda th: metrics(y, s, th)["F1"])
    return dict(experiment=name, notes=note, **{"AUC-ROC": roc_auc_score(y, s), "AUC-PR": average_precision_score(y, s)},
                **metrics(y, s, best))

# ---------- 1) Day-1 models (frozen, full, rules frozen, rules full, lookups)
t = pd.read_parquet(f"{D1}/test_preds.parquet")
rows = [visit_row(c[2:], t[c].values, t.CSN.values, t.y.values) for c in t.columns if c.startswith("p_")]

# ---------- 2) cell weights (frozen base + correction tables)
base = lgb.Booster(model_file=f"{D2}/base_frozen.txt")
cats = json.load(open(f"{D2}/base_frozen_categories.json"))
raw  = base.predict(apply_categories(t, cats, feats), raw_score=True)
for f in sorted(glob.glob(f"{D2}/weights_*_k*.csv")):
    kname = "prd_cell" if "prd_cell" in f else "cpt_plan"
    k = os.path.basename(f).split("_k")[1].replace(".csv", "")
    w = pd.read_csv(f, index_col="cell")["weight"]
    cell = t[KEYS[kname]].astype(str).agg("|".join, axis=1)
    rows.append(visit_row(f"cell weights {kname} k={k}", sig(raw + cell.map(w).fillna(0).values), t.CSN.values, t.y.values,
                          f"test lines with a weight: {cell.isin(w.index).mean():.1%}"))
met = pd.DataFrame(rows)
met.round(4).to_csv(f"experiment_log_metrics{TAG}.csv", index=False)

# ---------- 3) row counts (train / validation / update / test), same split as the runs
cols = ["CSN","GROUP_LABEL","SERVICEDATE","FILING_ORDER_C","FINAL_CPT","FINAL_MODIFIERS","PLACE_OF_SERVICE_CODE","PAYOR","PLAN"]
d = pe.add_dates(pb.load([2025, 2026], cols, SAMPLE, pafs.S3FileSystem()))
train, update, test = pe.split(d, "2026-02-01", "2026-05-01", "2026-08-01")
full = pd.concat([train, update], ignore_index=True)

def cnt(df):
    return dict(lines=len(df), visits=df.CSN.nunique(), rate=round(df.y.mean(), 4),
                period=f"{df.SVC_DATE.min():%Y-%m-%d} .. {df.SVC_DATE.max():%Y-%m-%d}")
def fit_val(df):
    a, b = next(GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42).split(df, groups=df.CSN))
    return cnt(df.iloc[a]), cnt(df.iloc[b])
fz, fl, te, up = fit_val(train), fit_val(full), cnt(test), cnt(update)

def counts(fit_val_pair, upd=None):
    fit, val = fit_val_pair
    return {"train_period": fit["period"], "train_lines": fit["lines"], "train_visits": fit["visits"], "train_pa_rate": fit["rate"],
            "val_lines": val["lines"], "val_visits": val["visits"], "val_pa_rate": val["rate"],
            "update_period": upd["period"] if upd else "", "update_lines": upd["lines"] if upd else 0,
            "update_visits": upd["visits"] if upd else 0, "update_pa_rate": upd["rate"] if upd else "",
            "test_period": te["period"], "test_lines": te["lines"], "test_visits": te["visits"], "test_pa_rate": te["rate"]}

base5, rules6 = "CPT, modifier, POS, payor, payor+plan", "CPT, modifier, POS, payor, payor+plan, rules"
info = {"lookup_frozen": ("lookup: PA rate per cell with fallback", fz, None),
        "lookup_full":   ("lookup: PA rate per cell with fallback", fl, None),
        "lgb_frozen": (base5, fz, None), "lgb_full": (base5, fl, None),
        "lgb_rules_frozen": (rules6, fz, None), "lgb_rules_full": (rules6, fl, None)}
for name in met.experiment:
    if name.startswith("cell weights"):
        kind = "CPT+payor+plan" if "cpt_plan" in name else "CPT+mod+POS+payor+plan"
        info[name] = (f"{rules6} + cell correction ({kind})", fz, up)

summary = pd.DataFrame([{"experiment": n, "features": info[n][0], **counts(info[n][1], info[n][2])} for n in met.experiment])
summary = summary.merge(met, on="experiment", how="left")
summary.insert(1, "sample", f"{SAMPLE:.0%} of visits")
summary.to_csv(f"experiment_summary{TAG}.csv", index=False)

pd.set_option("display.width", 300); pd.set_option("display.max_columns", 50)
print(met.round(4).to_string(index=False))
print(summary.T.to_string())
