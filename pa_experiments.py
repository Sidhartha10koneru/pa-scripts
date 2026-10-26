"""
pa_experiments.py - adaptation experiments for the PA model. Day 1: shared split + baselines.

Needs pa_base_adapter.py in the same folder (reuses its loading, cleaning, rules feature and metrics).

Shared split (by SERVICEDATE):
  TRAIN   2025 + early 2026 (everything before UPDATE)  -> frozen base models
  UPDATE  the 3 months before TEST (--update-months)     -> used by update methods (refit, cell weights...)
  TEST    last 3 months of data (--test-last-months)     -> scoring only, never used for fitting
  "full" models train on TRAIN + UPDATE = all data except TEST (the production-style model)
Look at the months first:  python pa_experiments.py months --sample 0.1

Mode "baselines" (Day 1) trains and scores on TEST:
  lookup_frozen      PA rate per cell with fallback (plan+CPT+mod+POS -> plan+CPT -> payor+CPT -> CPT), from TRAIN
  lookup_full        same, from TRAIN + UPDATE
  lgb_frozen         LightGBM on TRAIN only            (saved: the frozen base for later days)
  lgb_full           LightGBM on TRAIN + UPDATE = all non-test data (production-style; upper bound)
  (+ the same two with the rules feature if --rules-master/--rules-parquet are given)

Run:
  python pa_experiments.py baselines --demo                                   # synthetic check
  python pa_experiments.py baselines --sample 0.1 --out exp_d1 \
      --rules-master <...>/payor_master_review.xlsx --rules-parquet rules_analysis/all_rules.parquet

Outputs in --out: month_counts.csv, metrics.csv, calibration_bins.csv, operating_points.csv,
  test_preds.parquet, models/*.txt + models/*_categories.json, split_info.json
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit

import pa_base_adapter as pb
from pa_base_adapter import (CSN, DATE, CPT, MOD, POS, PAYOR_F, PLAN_F, BASE_FEATS, PARAMS,
                             fit_categories, apply_categories, scores, operating_points)

LOOKUP_LEVELS = [[CPT], [PAYOR_F, CPT], [PAYOR_F, PLAN_F, CPT], [PAYOR_F, PLAN_F, CPT, MOD, POS]]   # coarse -> fine, each level contains its parent
LOOKUP_M = 20          # smoothing: a cell needs ~20 cases before its own rate dominates its parent's


# ----------------------------------------------------------------- split
def add_dates(d):
    d["SVC_DATE"] = pd.to_datetime(d[DATE], errors="coerce")
    bad = d.SVC_DATE.isna().mean()
    if bad > 0.01:
        print(f"  WARNING: {bad:.1%} of {DATE} values could not be read as dates (dropped)")
    return d[d.SVC_DATE.notna()].copy()


def month_counts(d):
    m = d.assign(MONTH=d.SVC_DATE.dt.to_period("M").astype(str)).groupby("MONTH").agg(
        lines=("y", "size"), encounters=(CSN, "nunique"), pa_line_rate=("y", "mean"))
    m["pa_enc_rate"] = d.assign(MONTH=d.SVC_DATE.dt.to_period("M").astype(str)).groupby(["MONTH", CSN]).y.max() \
        .groupby("MONTH").mean()
    return m


def split(d, update_start, test_start, test_end=None):
    us, ts = pd.Timestamp(update_start), pd.Timestamp(test_start)
    te = pd.Timestamp(test_end) if test_end else d.SVC_DATE.max() + pd.Timedelta(days=1)
    train = d[d.SVC_DATE < us]
    update = d[(d.SVC_DATE >= us) & (d.SVC_DATE < ts)]
    test = d[(d.SVC_DATE >= ts) & (d.SVC_DATE < te)]
    # an encounter must not straddle windows: assign it to the window of its first service date
    first = d.groupby(CSN).SVC_DATE.min()
    win = pd.cut(first, [pd.Timestamp.min, us, ts, te], right=False, labels=["train", "update", "test"])
    train, update, test = (x[x[CSN].map(win) == w] for x, w in ((train, "train"), (update, "update"), (test, "test")))
    for n, x in (("TRAIN", train), ("UPDATE", update), ("TEST", test)):
        print(f"  {n:<6} {x.SVC_DATE.min():%Y-%m-%d} .. {x.SVC_DATE.max():%Y-%m-%d}: {len(x):,} lines, "
              f"{x[CSN].nunique():,} encounters, PA line rate {x.y.mean():.3%}")
    return train.reset_index(drop=True), update.reset_index(drop=True), test.reset_index(drop=True)


# ----------------------------------------------------------------- lookup baseline
def fit_lookup(df, m=LOOKUP_M):
    df = df.copy()
    keys = sorted({c for lv in LOOKUP_LEVELS for c in lv})
    df[keys] = df[keys].astype(str)
    g = df.y.mean()
    tabs, parent = [], None
    for cols in LOOKUP_LEVELS:
        t = df.groupby(cols, observed=True).y.agg(["sum", "count"]).reset_index()
        if parent is None:
            t["parent"] = g
        else:
            pcols, pt = parent
            t = t.merge(pt[pcols + ["p"]].rename(columns={"p": "parent"}), on=pcols, how="left")
            t["parent"] = t.parent.fillna(g)
        t["p"] = (t["sum"] + m * t.parent) / (t["count"] + m)
        tabs.append((cols, t))
        parent = (cols, t)
    return g, tabs


def predict_lookup(model, df):
    g, tabs = model
    x = df.copy()
    keys = sorted({c for lv in LOOKUP_LEVELS for c in lv})
    x[keys] = x[keys].astype(str)
    x["_row"] = np.arange(len(x))
    p = np.full(len(x), g)
    for cols, t in tabs:                                   # finer levels overwrite coarser ones when found
        hit = x[["_row"] + cols].merge(t[cols + ["p"]], on=cols, how="inner")
        p[hit._row.values] = hit.p.values
    return p


# ----------------------------------------------------------------- LightGBM
def train_lgb(df, feats, rounds=2000):
    cats = fit_categories(df, feats)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    tr_i, va_i = next(gss.split(df, groups=df[CSN]))
    tr, va = df.iloc[tr_i], df.iloc[va_i]
    stop = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(200)]
    m = lgb.train(PARAMS, lgb.Dataset(apply_categories(tr, cats, feats), tr.y), rounds,
                  valid_sets=[lgb.Dataset(apply_categories(va, cats, feats), va.y)], callbacks=stop)
    return m, cats


# ----------------------------------------------------------------- calibration
def calibration(y, p, n_bins=20):
    y, p = np.asarray(y, float), np.asarray(p, float)
    b = pd.qcut(pd.Series(p).rank(method="first"), n_bins, labels=False)
    t = pd.DataFrame({"y": y, "p": p, "b": b}).groupby("b").agg(n=("y", "size"), pred=("p", "mean"), actual=("y", "mean"))
    ece = (t.n * (t.pred - t.actual).abs()).sum() / t.n.sum()
    return {"brier": np.mean((p - y) ** 2), "ECE": ece, "mean_pred": p.mean(), "actual_rate": y.mean()}, t


def evaluate(name, test, p, rows, cal_rows, ops):
    s = scores(test.y.values, p, test[CSN].values)
    c_line, bins_line = calibration(test.y.values, p)
    e = pd.DataFrame({"y": test.y.values, "p": p, "c": test[CSN].values}).groupby("c").max()
    c_enc, bins_enc = calibration(e.y.values, e.p.values)
    rows.append({"model": name, **s, **{f"{k}_line": v for k, v in c_line.items()},
                 **{f"{k}_enc": v for k, v in c_enc.items()}})
    cal_rows += [bins_line.assign(model=name, unit="line").reset_index(),
                 bins_enc.assign(model=name, unit="encounter").reset_index()]
    ops += operating_points(test.y.values, p, "line", name) + operating_points(e.y.values, e.p.values, "encounter", name)


# ----------------------------------------------------------------- Day 1
def baselines(train, update, test, out, rules):
    os.makedirs(f"{out}/models", exist_ok=True)
    full = pd.concat([train, update], ignore_index=True)
    rows, cal_rows, ops, preds = [], [], [], {}

    for tag, data in (("frozen", train), ("full", full)):
        print(f"\n== lookup_{tag}")
        preds[f"lookup_{tag}"] = predict_lookup(fit_lookup(data), test)

    sets = {"lgb": BASE_FEATS}
    if rules:
        sets["lgb_rules"] = BASE_FEATS + ["RULE_PLAN_SHARE"]
    for sname, feats in sets.items():
        for tag, data in (("frozen", train), ("full", full)):
            name = f"{sname}_{tag}"
            t0 = time.time()
            print(f"\n== {name}: LightGBM on {len(data):,} lines, features {feats}")
            m, cats = train_lgb(data, feats)
            m.save_model(f"{out}/models/{name}.txt")
            json.dump(cats, open(f"{out}/models/{name}_categories.json", "w"))
            preds[name] = m.predict(apply_categories(test, cats, feats))
            print(f"   {m.num_trees()} trees, {time.time()-t0:.0f}s")

    for name, p in preds.items():
        evaluate(name, test, p, rows, cal_rows, ops)

    met = pd.DataFrame(rows).set_index("model")
    met.to_csv(f"{out}/metrics.csv")
    pd.concat(cal_rows).to_csv(f"{out}/calibration_bins.csv", index=False)
    pd.DataFrame(ops).to_csv(f"{out}/operating_points.csv", index=False)
    keep = [CSN, "SVC_DATE", CPT, MOD, POS, PAYOR_F, PLAN_F, "y"] + (["RULE_PLAN_SHARE"] if rules else [])
    test[keep].assign(**{f"p_{k}": v for k, v in preds.items()}).to_parquet(f"{out}/test_preds.parquet", index=False)

    pd.set_option("display.width", 250)
    cols = ["AP_enc", "AUC_enc", "AP_line", "AP_lift_line", "brier_enc", "ECE_enc", "mean_pred_enc", "actual_rate_enc"]
    print("\n== BASELINES on TEST\n", met[cols].round(4).to_string())
    o = pd.DataFrame(ops)
    o = o[o.unit == "encounter"]
    req = o[o.direction == "PA required"].pivot(index="model", columns="target", values="PA_caught (recall)")
    clr = o[(o.direction == "PA not required") & (o.target == 0.995)].set_index("model")[
        ["share_auto_cleared", "PA_wrongly_cleared"]]
    print("\n== Encounter: share of PA caught at 90% / 94% precision\n", req.round(4).to_string())
    print("\n== Encounter: at 99.5% NPV, share auto-cleared and share of PA wrongly cleared\n", clr.round(4).to_string())
    print(f"\nSaved to {out}/")


# ----------------------------------------------------------------- main
def demo_data():
    tr, te, _ = pb.demo()
    d = pd.concat([tr, te], ignore_index=True)
    rng = np.random.default_rng(1)
    enc_month = pd.Series(rng.integers(1, 13, d[CSN].nunique()), index=d[CSN].unique())
    d[DATE] = pd.to_datetime(dict(year=d.YEAR, month=d[CSN].map(enc_month), day=15))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["months", "baselines"],
                    help="months = only show lines/PA rate per month (no training); baselines = Day 1 run")
    ap.add_argument("--train-years", nargs="+", type=int, default=[2025])
    ap.add_argument("--test-years", nargs="+", type=int, default=[2026])
    ap.add_argument("--update-start", default=None, help="explicit update-window start (overrides --update-months)")
    ap.add_argument("--update-months", type=int, default=3,
                    help="UPDATE = the N months just before TEST (default 3); everything earlier is TRAIN")
    ap.add_argument("--test-start", default=None, help="explicit test start date (overrides --test-last-months)")
    ap.add_argument("--test-last-months", type=int, default=3,
                    help="TEST = the last N calendar months present in the data (default 3)")
    ap.add_argument("--test-end", default=None)
    ap.add_argument("--sample", type=float, default=0.1)
    ap.add_argument("--rules-master"); ap.add_argument("--rules-parquet")
    ap.add_argument("--out", default="exp_d1")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()

    if a.demo:
        d = demo_data()
        a.out = a.out + "_demo"
        rules = True
    else:
        import pyarrow.fs as pafs
        fs = pafs.S3FileSystem()
        raw = [c for c in BASE_FEATS if c not in (PAYOR_F, PLAN_F)] + [pb.PAYOR_NAME, pb.PLAN_NAME]
        cols = list(dict.fromkeys([CSN, pb.LABEL, DATE, pb.FILING] + raw))
        print("Loading", a.train_years + a.test_years)
        d = pb.load(sorted(set(a.train_years + a.test_years)), cols, a.sample, fs)
        rules = bool(a.rules_master and a.rules_parquet)
        if rules:
            d = pb.add_rules_feature(d, a.rules_master, a.rules_parquet)

    os.makedirs(a.out, exist_ok=True)
    d = add_dates(d)
    mc = month_counts(d)
    mc.to_csv(f"{a.out}/month_counts.csv")
    print("\n== Lines and PA rate by month\n", mc.round(4).to_string())
    if a.mode == "months":
        return
    if a.test_start is None:                                     # last N calendar months in the data
        last = d.SVC_DATE.max().to_period("M")
        a.test_start = str((last - (a.test_last_months - 1)).start_time.date())
        print(f"\nTEST = last {a.test_last_months} months: {a.test_start} .. {d.SVC_DATE.max():%Y-%m-%d}"
              f"  (last month in data: {last}; check it is not a partial month)")
    if a.update_start is None:
        a.update_start = str((pd.Timestamp(a.test_start).to_period("M") - a.update_months).start_time.date())
    print(f"TRAIN  = everything before {a.update_start}  (2025 + early 2026)")
    print(f"UPDATE = {a.update_start} .. {a.test_start}   (used only by update methods; the 'full' models train on TRAIN+UPDATE)")

    print("\n== Split")
    train, update, test = split(d, a.update_start, a.test_start, a.test_end)
    json.dump({"update_start": a.update_start, "test_start": a.test_start, "test_end": a.test_end,
               "sample": a.sample, "rules": rules,
               "rows": {"train": len(train), "update": len(update), "test": len(test)}},
              open(f"{a.out}/split_info.json", "w"), indent=1)
    if min(len(train), len(update), len(test)) == 0:
        raise SystemExit("One window is empty - change --update-start / --test-start.")
    baselines(train, update, test, a.out, rules)


if __name__ == "__main__":
    main()
