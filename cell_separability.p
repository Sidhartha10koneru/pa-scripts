"""
cell_separability.py - for the biggest mixed cells (bucket 4): can ANY pre-visit information separate
the PA visits from the non-PA visits inside the cell?  One general test, no pattern list.

Per cell (5-feature cell: CPT + modifier + POS + payor|plan):
  A. label checks   [V4, incl. AMBIGUOUS]: PA rate by year, AMBIGUOUS share by year, inflation ratio
                    (rate as labelled / rate if AMBIGUOUS = FALSE), source of TRUE (LEVEL1/LEVEL2), blank-auth FALSE
  separability      [V4, labelled lines]: a small LightGBM trained ONLY on this cell, on
                    (a) every column known before the visit (leak / ID / name / date columns removed) and
                    (b) generic history features built from EARLIER dates only:
                        - visit number this calendar year for the patient: in this cell, and in this plan's
                          therapy codes (PT/OT/speech/chiro) when the cell is a therapy code
                        - patient had a PA line in this cell more than LAG days before
                        - days since the patient's first visit in the cell, month of year, age band
                    trained on lines before TEST_START, scored on the test window (forward in time).
Category (primary; A is a separate flag):
  D  data insufficient   too few PA lines to fit or to judge
  B  fixable by features out-of-time AUC >= --auc-ok   (top features say which)
  C  recalibration       not separable, but the cell's PA rate shifted between training and test
  E  not separable        nothing available separates PA from non-PA -> needs a new data source
  A_flag                 label rule behaves differently by year (inflation) -> check labels first

Inputs : exp_nofilter_2022/fn_mixed_cells_recall80.csv  (cell 9 of the bucket notebook)
Outputs: <out>/cells.csv (one row per cell), <out>/rollup.csv (FN by category), <out>/features_<n>.csv
Run    : python cell_separability.py --cells exp_nofilter_2022/fn_mixed_cells_recall80.csv --cover 0.5 --out sep_mixed
"""
import argparse
import os
import re

import numpy as np
import pandas as pd

import pa_base_adapter as pb

KEY = ["FINAL_CPT", "FINAL_MODIFIERS", "PLACE_OF_SERVICE_CODE", "PAYOR_PLAN_CLEAN"]
THERAPY = re.compile(r"^(97\d{3}|9894[0-2]|92507|92526)$")
EXCLUDE = re.compile(
    r"LABEL|^N_(TRUE|FALSE|AMBIGUOUS)$|AUTH|CARC|RARC|REMIT|DENI|PAID|PAYMENT|ALLOW|ADJUST|CLAIM|GUID|GROUP_SIZE|"
    r"^CSN|PAT_ID|MRN|ACCOUNT|_NAME$|NAME$|FIRST|LAST|BIRTH|ZIP|ADDRESS|PHONE|SSN|DATE|^y$|^L$|SVC|PAYOR$|PLAN$|"
    r"PAYOR_CLEAN|PAYOR_PLAN_CLEAN|FINAL_CPT|FINAL_MODIFIERS|PLACE_OF_SERVICE_CODE", re.I)


def fs_():
    import pyarrow.fs as pafs
    return pafs.S3FileSystem()


def load(cpts, plans, years, fs):
    """[V4] all columns, all labels (incl. AMBIGUOUS), rows of these CPTs and plans only"""
    import pyarrow.dataset as ds
    parts = []
    for y in years:
        dset = ds.dataset(pb.BASE_PATH.format(year=y).replace("s3://", ""), filesystem=fs, format="parquet")
        t = dset.to_table(filter=ds.field("FINAL_CPT").isin(sorted(cpts))).to_pandas()
        t = pb.clean_names(t.loc[:, ~t.columns.duplicated()])
        t = t[t.PAYOR_PLAN_CLEAN.isin(plans)]
        parts.append(t)
        print(f"  [V4] {y}: {len(t):,} rows for {len(cpts)} CPTs / {len(plans)} plans")
    d = pd.concat(parts, ignore_index=True)
    for c in KEY:
        d[c] = d[c].astype(object).fillna("").astype(str).str.strip()
    d["L"] = d.GROUP_LABEL.astype(str).str.strip().str.upper().map(
        {"TRUE": "TRUE", "1": "TRUE", "FALSE": "FALSE", "0": "FALSE"}).fillna("AMBIGUOUS")
    d["SVC"] = pd.to_datetime(d.SERVICEDATE, errors="coerce")
    return d[d.SVC.notna()].copy()


def dedup(x):
    keys = [c for c in ["CSN", "FINAL_CPT", "FINAL_MODIFIERS", "SERVICEDATE", "FILING_ORDER_C"] if c in x]
    return x.assign(_t=(x.L == "TRUE")).sort_values("_t", ascending=False).drop_duplicates(keys).drop(columns="_t")


def label_checks(c_all):
    lab = c_all[c_all.L.isin(["TRUE", "FALSE"])]
    out = {}
    for y, g in c_all.groupby(c_all.SVC.dt.year):
        n_lab = g.L.isin(["TRUE", "FALSE"]).sum()
        out[f"rate_{y}"] = (g.L == "TRUE").sum() / n_lab if n_lab else np.nan
        out[f"amb_share_{y}"] = (g.L == "AMBIGUOUS").mean()
        out[f"lines_{y}"] = int(n_lab)
    lb = (c_all.L == "TRUE").sum() / max(len(c_all), 1)
    out["rate_labelled"] = (lab.L == "TRUE").mean() if len(lab) else np.nan
    out["rate_if_amb_false"] = lb
    out["inflation_x"] = out["rate_labelled"] / lb if lb > 0 else np.nan
    t = lab[lab.L == "TRUE"]
    if len(t) and "LEVEL1_LABEL" in t:
        l1 = t.LEVEL1_LABEL.astype(str).eq("TRUE")
        l2 = t.get("LEVEL2_LABEL", pd.Series("", index=t.index)).astype(str).eq("TRUE")
        out["true_from_auth"] = round(l1.mean(), 3)
        out["true_from_code"] = round((~l1 & l2).mean(), 3)
    f = lab[lab.L == "FALSE"]
    if len(f) and "LINE_AUTHORIZATION_NUMBER" in f:
        a = f.LINE_AUTHORIZATION_NUMBER.astype(object).fillna("").astype(str).str.strip()
        out["false_blank_auth"] = round(a.isin(["", "nan", "None"]).mean(), 3)
    return out


def history(x, plan_rows, lag_days):
    """features from EARLIER dates only (same patient)"""
    x = x.sort_values(["PAT_ID", "SVC"]).copy()
    yr = x.SVC.dt.year
    day = x[["PAT_ID", "SVC"]].assign(yr=yr).drop_duplicates()
    day["cell_visit_no_ytd"] = day.groupby(["PAT_ID", "yr"]).cumcount() + 1
    x = x.merge(day[["PAT_ID", "SVC", "cell_visit_no_ytd"]], on=["PAT_ID", "SVC"], how="left")
    first = x.groupby("PAT_ID").SVC.transform("min")
    x["days_since_first_cell_visit"] = (x.SVC - first).dt.days
    x["month"] = x.SVC.dt.month
    # prior PA in this cell, older than lag (claims arrive late)
    pa = x[x.y == 1][["PAT_ID", "SVC"]].sort_values("SVC").rename(columns={"SVC": "pa_date"})
    x = x.sort_values("SVC")
    x["_cut"] = x.SVC - pd.Timedelta(days=lag_days)
    m = pd.merge_asof(x.sort_values("_cut"), pa, left_on="_cut", right_on="pa_date", by="PAT_ID", direction="backward")
    x = m.drop(columns=["_cut"])
    x["prior_pa_lagged"] = x.pa_date.notna().astype(int)
    x = x.drop(columns=["pa_date"])
    if plan_rows is not None and len(plan_rows):          # therapy: visits in this plan, any therapy code, this year
        v = plan_rows[["PAT_ID", "SVC"]].drop_duplicates().sort_values(["PAT_ID", "SVC"])
        v["yr"] = v.SVC.dt.year
        v["therapy_visit_no_ytd"] = v.groupby(["PAT_ID", "yr"]).cumcount() + 1
        x = x.merge(v[["PAT_ID", "SVC", "therapy_visit_no_ytd"]], on=["PAT_ID", "SVC"], how="left")
    if "BIRTH_DATE" in x:
        age = (x.SVC - pd.to_datetime(x.BIRTH_DATE, errors="coerce")).dt.days / 365.25
        x["age_band"] = pd.cut(age, [-1, 18, 40, 65, 75, 200], labels=["<18", "18-39", "40-64", "65-74", "75+"]).astype(str)
    return x


def separability(x, test_start, min_pa_fit, min_pa_eval):
    import lightgbm as lgb
    from sklearn.metrics import average_precision_score, roc_auc_score
    fit, ev = x[x.SVC < test_start], x[x.SVC >= test_start]
    res = {"fit_lines": len(fit), "fit_pa": int(fit.y.sum()), "eval_lines": len(ev), "eval_pa": int(ev.y.sum()),
           "train_rate": fit.y.mean() if len(fit) else np.nan, "test_rate": ev.y.mean() if len(ev) else np.nan}
    if res["fit_pa"] < min_pa_fit or res["eval_pa"] < min_pa_eval or fit.y.nunique() < 2 or ev.y.nunique() < 2:
        return res, None
    hist = [c for c in ["cell_visit_no_ytd", "therapy_visit_no_ytd", "days_since_first_cell_visit", "month",
                        "prior_pa_lagged", "age_band"] if c in x]
    cols = []
    for c in x.columns:
        if c in hist or EXCLUDE.search(c):
            continue
        nu = fit[c].nunique(dropna=True)
        if nu < 2 or nu > 0.5 * len(fit):                 # constant, or ID-like
            continue
        cols.append(c)
    feats = cols + hist
    X = pd.DataFrame(index=x.index)
    for c in feats:
        if c in hist and c != "age_band":
            X[c] = pd.to_numeric(x[c], errors="coerce")
        else:
            v = x[c].astype(object).fillna("NA").astype(str)
            keep = v[x.SVC < test_start].value_counts()
            keep = set(keep[keep >= 5].index)
            X[c] = pd.Categorical(v.where(v.isin(keep), "OTHER"))
    p = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=20, feature_fraction=0.8,
             bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=42)
    fi, ei = (x.SVC < test_start).values, (x.SVC >= test_start).values
    m = lgb.train(p, lgb.Dataset(X[fi], x.y[fi]), 300)
    s = m.predict(X[ei])
    yv = x.y[ei].values
    res |= {"AUC_oot": roc_auc_score(yv, s), "AP_oot": average_precision_score(yv, s), "n_features": len(feats)}
    o = np.argsort(-s)
    tp = np.cumsum(yv[o]); prec = tp / np.arange(1, len(s) + 1)
    res["recall_at_50pct_precision"] = float(tp[prec >= 0.5].max() / yv.sum()) if (prec >= 0.5).any() else 0.0
    imp = pd.Series(m.feature_importance("gain"), index=feats).sort_values(ascending=False)
    imp = imp / imp.sum() if imp.sum() else imp
    res["top_features"] = "; ".join(f"{k} {v:.2f}" for k, v in imp.head(5).items())
    return res, imp


def categorise(r, a):
    if not np.isfinite(r.get("AUC_oot", np.nan)):
        cat = "D data insufficient"
    elif r["AUC_oot"] >= a.auc_ok:
        cat = "B fixable by features"
    elif np.isfinite(r["train_rate"]) and r["train_rate"] > 0 and (
            r["test_rate"] / r["train_rate"] >= a.shift_x or r["test_rate"] / r["train_rate"] <= 1 / a.shift_x):
        cat = "C recalibration (rate shifted)"
    else:
        cat = "E not separable with current data"
    flag = bool(np.isfinite(r.get("inflation_x", np.nan)) and r["inflation_x"] >= a.inflation_x)
    return cat, flag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="exp_nofilter_2022/fn_mixed_cells_recall80.csv")
    ap.add_argument("--cover", type=float, default=0.5, help="take the biggest cells until this share of bucket-4 FN")
    ap.add_argument("--max-cells", type=int, default=80)
    ap.add_argument("--years", nargs="+", type=int, default=[2025, 2026])
    ap.add_argument("--test-start", default="2026-05-01")
    ap.add_argument("--test-end", default="2026-08-01")
    ap.add_argument("--lag-days", type=int, default=45)
    ap.add_argument("--min-pa-fit", type=int, default=15)
    ap.add_argument("--min-pa-eval", type=int, default=10)
    ap.add_argument("--auc-ok", type=float, default=0.75)
    ap.add_argument("--shift-x", type=float, default=2.0)
    ap.add_argument("--inflation-x", type=float, default=2.0)
    ap.add_argument("--out", default="sep_mixed")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 80)

    cells = pd.read_csv(a.cells, dtype={k: str for k in KEY})
    for c in KEY:
        cells[c] = cells[c].fillna("").astype(str).str.strip()
    cells = cells.sort_values("fn_services", ascending=False)
    share = cells.fn_services.cumsum() / cells.fn_services.sum()
    sel = cells[(share.shift(fill_value=0) < a.cover)].head(a.max_cells).reset_index(drop=True)
    print(f"{len(sel)} cells selected, {sel.fn_services.sum():,} of {cells.fn_services.sum():,} bucket-4 FN "
          f"({sel.fn_services.sum() / cells.fn_services.sum():.0%})")

    cpts = set(sel.FINAL_CPT)
    plans = set(sel.PAYOR_PLAN_CLEAN)
    if any(THERAPY.match(c) for c in cpts):
        cpts |= {c for c in ["97110", "97112", "97116", "97140", "97530", "97535", "97161", "97162", "97163", "97164",
                             "97165", "97166", "97167", "97168", "98940", "98941", "98942", "92507", "92526"]}
    d = load(cpts, plans, a.years, fs_())
    d = d[d.SVC < pd.Timestamp(a.test_end)]

    rows = []
    for i, c in sel.iterrows():
        m = np.logical_and.reduce([d[k] == c[k] for k in KEY])
        c_all = d[m]
        r = {**{k: c[k] for k in KEY}, "fn_services": int(c.fn_services), **label_checks(c_all)}
        x = dedup(c_all[c_all.L.isin(["TRUE", "FALSE"])]).copy()
        x["y"] = (x.L == "TRUE").astype(int)
        plan_rows = None
        if THERAPY.match(c.FINAL_CPT):
            plan_rows = d[(d.PAYOR_PLAN_CLEAN == c.PAYOR_PLAN_CLEAN) & d.FINAL_CPT.str.match(THERAPY.pattern)]
        if "PAT_ID" in x and len(x):
            x = history(x, plan_rows, a.lag_days)
        s, imp = separability(x, pd.Timestamp(a.test_start), a.min_pa_fit, a.min_pa_eval) if len(x) else ({}, None)
        r |= s
        r["category"], r["A_label_flag"] = categorise(r, a)
        if imp is not None:
            imp.rename("gain_share").to_csv(f"{a.out}/features_{i:03d}.csv")
        rows.append(r)
        print(f"[{i + 1}/{len(sel)}] {c.FINAL_CPT} {c.FINAL_MODIFIERS or '-'} {c.PLACE_OF_SERVICE_CODE} "
              f"{c.PAYOR_PLAN_CLEAN[:45]:<45} FN {int(c.fn_services):>4} | {r['category']}"
              f"{' + A' if r['A_label_flag'] else ''} | AUC {r.get('AUC_oot', np.nan):.2f} | {r.get('top_features', '')[:70]}")

    out = pd.DataFrame(rows)
    out.to_csv(f"{a.out}/cells.csv", index=False)
    roll = out.groupby("category").agg(cells=("fn_services", "size"), fn_services=("fn_services", "sum"))
    roll["share_of_selected_fn"] = (roll.fn_services / roll.fn_services.sum()).round(3)
    roll["with_label_flag"] = out.groupby("category").A_label_flag.sum()
    roll.to_csv(f"{a.out}/rollup.csv")
    print("\n== rollup (bucket-4 FN in the selected cells)\n", roll.to_string())
    print(f"\ncells with label flag (A): {int(out.A_label_flag.sum())} | outputs in {a.out}/")


if __name__ == "__main__":
    main()

#nohup python -u cell_separability.py --cells exp_nofilter_2022/fn_mixed_cells_recall80.csv --cover 0.5 --out sep_mixed > run_sep_mixed.log 2>&1 &
