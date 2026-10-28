import pandas as pd, numpy as np, pyarrow.fs as pafs
from sklearn.model_selection import GroupShuffleSplit
import pa_base_adapter as pb, pa_experiments as pe

# 1) same data + same windows as the runs (rules don't change row counts, so not needed here)
cols = ["CSN","GROUP_LABEL","SERVICEDATE","FILING_ORDER_C","FINAL_CPT","FINAL_MODIFIERS","PLACE_OF_SERVICE_CODE","PAYOR","PLAN"]
d = pe.add_dates(pb.load([2025, 2026], cols, 0.3, pafs.S3FileSystem()))
train, update, test = pe.split(d, "2026-02-01", "2026-05-01", "2026-08-01")
full = pd.concat([train, update], ignore_index=True)

def cnt(df):
    return dict(lines=len(df), visits=df.CSN.nunique(), pa_line_rate=round(df.y.mean(), 4),
                period=f"{df.SVC_DATE.min():%Y-%m-%d} .. {df.SVC_DATE.max():%Y-%m-%d}")

def fit_val(df):                       # same 90/10 split by visit as train_lgb (seed 42)
    tr_i, va_i = next(GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42).split(df, groups=df.CSN))
    return cnt(df.iloc[tr_i]), cnt(df.iloc[va_i])

fz_fit, fz_val = fit_val(train)
fl_fit, fl_val = fit_val(full)
te, up = cnt(test), cnt(update)

def row(name, features, fit, val, upd=None):
    r = {"experiment": name, "features": features,
         "train_period": fit["period"], "train_lines": fit["lines"], "train_visits": fit["visits"], "train_pa_rate": fit["pa_line_rate"],
         "val_lines": val["lines"], "val_visits": val["visits"], "val_pa_rate": val["pa_line_rate"],
         "update_period": upd["period"] if upd else "", "update_lines": upd["lines"] if upd else 0,
         "update_visits": upd["visits"] if upd else 0, "update_pa_rate": upd["pa_line_rate"] if upd else "",
         "test_period": te["period"], "test_lines": te["lines"], "test_visits": te["visits"], "test_pa_rate": te["pa_line_rate"]}
    return r

base5, rules6 = "CPT, modifier, POS, payor, payor+plan", "CPT, modifier, POS, payor, payor+plan, rules"
sheet = pd.DataFrame([
    row("lgb_frozen",        base5,  fz_fit, fz_val),
    row("lgb_full",          base5,  fl_fit, fl_val),
    row("lgb_rules_frozen",  rules6, fz_fit, fz_val),
    row("lgb_rules_full",    rules6, fl_fit, fl_val),
    row("cell weights cpt_plan k=20", rules6 + " + cell correction (CPT+payor+plan)", fz_fit, fz_val, up),
    row("cell weights prd_cell k=20", rules6 + " + cell correction (CPT+mod+POS+payor+plan)", fz_fit, fz_val, up),
])

# 2) attach the metrics already computed
m1 = pd.read_csv("experiment_log_metrics.csv").query("run == 'exp_d1_s30'").drop(columns="run")
m2 = pd.read_csv("experiment_log_cell_weights.csv").drop(columns="notes")
m2["model"] = m2.model.str.replace(r"cell weights (\w+) k=(\d+)", r"cell weights \1 k=\2", regex=True)
metrics = pd.concat([m1, m2]).rename(columns={"model": "experiment"})
sheet = sheet.merge(metrics, on="experiment", how="left")

sheet.to_csv("experiment_summary.csv", index=False)
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 50)
print(sheet.T.to_string())          # transposed so it fits on screen
