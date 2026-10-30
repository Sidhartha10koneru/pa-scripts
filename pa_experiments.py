"""
pa_experiments.py - adaptation experiments for the PA model. Day 1: shared split + baselines.

Needs pa_base_adapter.py in the same folder (reuses its loading, cleaning, rules feature and metrics). NEW_second

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
from sklearn.metrics import average_precision_score, roc_auc_score

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
    d = d[d.SVC_DATE.notna()].copy()
    pb.log_step("all", "5. unreadable service date dropped", d)
    return d


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
        pb.log_step(n, "6. window (visit kept in window of its first date)", x)
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
def split_stats(df, prefix=""):
    """train / validation sizes exactly as train_lgb splits them (10% of visits, seed 42)"""
    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    tr_i, va_i = next(gss.split(df, groups=df[CSN]))
    out = {}
    for n, x in (("train", df.iloc[tr_i]), ("val", df.iloc[va_i])):
        out |= {f"{n}_lines": len(x), f"{n}_visits": int(x[CSN].nunique()), f"{n}_pa_rate": float(x.y.mean()),
                f"{n}_start": str(x.SVC_DATE.min().date()), f"{n}_end": str(x.SVC_DATE.max().date())}
    return out


def window_stats(x, n):
    return {f"{n}_lines": len(x), f"{n}_visits": int(x[CSN].nunique()), f"{n}_pa_rate": float(x.y.mean()) if len(x) else None,
            f"{n}_period": f"{x.SVC_DATE.min():%Y-%m-%d} .. {x.SVC_DATE.max():%Y-%m-%d}" if len(x) else ""}


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


def service_level(test, p):
    """service = CSN + CPT: one row per distinct service in a visit (label = any copy TRUE, score = max)"""
    return pd.DataFrame({"y": test.y.values, "p": p, "c": test[CSN].values, "cpt": test[CPT].values}) \
        .groupby(["c", "cpt"], sort=False)[["y", "p"]].max()


def evaluate(name, test, p, rows, cal_rows, ops):
    s = scores(test.y.values, p, test[CSN].values)
    sv = service_level(test, p)
    s |= {"services": len(sv), "PA_services": int(sv.y.sum()), "pa_service_rate": sv.y.mean(),
          "AP_svc": average_precision_score(sv.y, sv.p), "AUC_svc": roc_auc_score(sv.y, sv.p)}
    c_svc, bins_svc = calibration(sv.y.values, sv.p.values)
    c_line, bins_line = calibration(test.y.values, p)
    e = pd.DataFrame({"y": test.y.values, "p": p, "c": test[CSN].values}).groupby("c").max()
    c_enc, bins_enc = calibration(e.y.values, e.p.values)
    rows.append({"model": name, **s, **{f"{k}_svc": v for k, v in c_svc.items()},
                 **{f"{k}_line": v for k, v in c_line.items()},
                 **{f"{k}_enc": v for k, v in c_enc.items()}})
    cal_rows += [bins_line.assign(model=name, unit="line").reset_index(),
                 bins_enc.assign(model=name, unit="encounter").reset_index(),
                 bins_svc.assign(model=name, unit="service").reset_index()]
    ops += operating_points(sv.y.values, sv.p.values, "service", name) + recall_points(sv.y.values, sv.p.values, "service", name)
    ops += recall_points(e.y.values, e.p.values, "encounter", name)
    ops += operating_points(test.y.values, p, "line", name) + operating_points(e.y.values, e.p.values, "encounter", name)


def recall_points(y, p, unit, model, targets=(0.8, 0.9, 0.95)):
    """confusion matrix at the cut-off that catches 80/90/95% of PA"""
    y, p = np.asarray(y), np.asarray(p)
    rows = []
    for r in targets:
        cut = np.quantile(p[y == 1], 1 - r)
        f = p >= cut
        tp, fp = int((f & (y == 1)).sum()), int((f & (y == 0)).sum())
        fn, tn = int((~f & (y == 1)).sum()), int((~f & (y == 0)).sum())
        rows.append({"unit": unit, "model": model, "direction": "recall target", "target": r, "threshold": cut,
                     "share_flagged": f.mean(), "PA_caught (recall)": tp / max(tp + fn, 1),
                     "achieved": tp / max(tp + fp, 1), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                     "F1": 2 * tp / max(2 * tp + fp + fn, 1)})
    return rows


# ----------------------------------------------------------------- Day 1
def add_dx(d, extra):
    """Clean Primary_DX and add DX3 (first 3 characters = ICD-10 category) so rare codes share information."""
    if "Primary_DX" in extra or "DX3" in extra:
        dx = d["Primary_DX"].astype(str).str.upper().str.replace(".", "", regex=False).str.strip()
        d["Primary_DX"] = dx.where(~dx.isin(["", "NAN", "NONE", "NULL"]))
        d["DX3"] = d["Primary_DX"].str[:3]
        print(f"  Primary_DX filled on {d.Primary_DX.notna().mean():.1%} of lines; "
              f"{d.Primary_DX.nunique():,} codes, {d.DX3.nunique():,} 3-char categories")
    return d


def baselines(train, update, test, out, rules, extra=(), only=None):
    os.makedirs(f"{out}/models", exist_ok=True)
    full = pd.concat([train, update], ignore_index=True)
    rows, cal_rows, ops, preds = [], [], [], {}

    want = (lambda n: only is None or n in only)
    for tag, data in (("frozen", train), ("full", full)):
        if not want(f"lookup_{tag}"):
            continue
        print(f"\n== lookup_{tag}")
        preds[f"lookup_{tag}"] = predict_lookup(fit_lookup(data), test)

    sets = {"lgb": BASE_FEATS}
    if rules:
        sets["lgb_rules"] = BASE_FEATS + ["RULE_PLAN_SHARE"]
    if extra:
        ctx = list(extra) + (["DX3"] if "Primary_DX" in extra else [])   # "DX3" alone = category only
        sets["lgb_rules_ctx" if rules else "lgb_ctx"] = sets.get("lgb_rules", BASE_FEATS) + ctx
    for sname, feats in sets.items():
        for tag, data in (("frozen", train), ("full", full)):
            name = f"{sname}_{tag}"
            if not want(name):
                continue
            t0 = time.time()
            print(f"\n== {name}: LightGBM on {len(data):,} lines, features {feats}")
            m, cats = train_lgb(data, feats)
            m.save_model(f"{out}/models/{name}.txt")
            json.dump(cats, open(f"{out}/models/{name}_categories.json", "w"))
            preds[name] = m.predict(apply_categories(test, cats, feats))
            print(f"   {m.num_trees()} trees, {time.time()-t0:.0f}s")

    for name, p in preds.items():
        evaluate(name, test, p, rows, cal_rows, ops)

    # sizes for the experiment sheet: train/validation as split inside training, update and test windows
    st = {"frozen": split_stats(train), "full": split_stats(full)}
    counts = {name: {**st["full" if name.endswith("_full") else "frozen"], **window_stats(test, "test"),
                     "update_lines": 0, "update_visits": 0, "update_pa_rate": None, "update_period": ""}
              for name in preds}
    json.dump(counts, open(f"{out}/split_counts.json", "w"), indent=1)

    met = pd.DataFrame(rows).set_index("model")
    met.to_csv(f"{out}/metrics.csv")
    pd.concat(cal_rows).to_csv(f"{out}/calibration_bins.csv", index=False)
    pd.DataFrame(ops).to_csv(f"{out}/operating_points.csv", index=False)
    keep = [CSN, "SVC_DATE", CPT, MOD, POS, PAYOR_F, PLAN_F, "y"] + (["RULE_PLAN_SHARE"] if rules else []) + \
        [c for c in ("Primary_DX", "DX3") if c in test.columns]
    test[keep].assign(**{f"p_{k}": v for k, v in preds.items()}).to_parquet(f"{out}/test_preds.parquet", index=False)

    pd.set_option("display.width", 250)
    cols = ["AP_svc", "AUC_svc", "PA_services", "services", "AP_enc", "AUC_enc", "AP_line", "AP_lift_line", "brier_enc", "ECE_enc", "mean_pred_enc", "actual_rate_enc"]
    print("\n== BASELINES on TEST\n", met[cols].round(4).to_string())
    o = pd.DataFrame(ops)
    o = o[o.unit == "encounter"]
    req = o[o.direction == "PA required"].pivot(index="model", columns="target", values="PA_caught (recall)")
    clr = o[(o.direction == "PA not required") & (o.target == 0.995)].set_index("model")[
        ["share_auto_cleared", "PA_wrongly_cleared"]]
    print("\n== Encounter: share of PA caught at 90% / 94% precision\n", req.round(4).to_string())
    print("\n== Encounter: at 99.5% NPV, share auto-cleared and share of PA wrongly cleared\n", clr.round(4).to_string())
    r = pd.DataFrame(ops)
    r = r[r.direction == "recall target"]
    print("\n== Confusion matrix at 80/90/95% recall (service = CSN+CPT, encounter = CSN)\n",
          r[["unit", "model", "target", "threshold", "share_flagged", "achieved", "TP", "FP", "FN", "TN", "F1"]]
          .rename(columns={"achieved": "precision"}).round(4).to_string(index=False))
    print(f"\nSaved to {out}/")



# ----------------------------------------------------------------- Day 2: cell weights (Task 1)
CELL_KEYS = {"cpt_plan": [CPT, PLAN_F], "prd_cell": [CPT, MOD, POS, PLAN_F]}


def logit(x):
    x = np.clip(np.asarray(x, float), 0.002, 0.998)
    return np.log(x / (1 - x))


def sig(z):
    return 1 / (1 + np.exp(-np.asarray(z, float)))


def cell_key(df, key):
    return df[key].astype(str).agg("|".join, axis=1)


def learn_weights(df, raw, key, k):
    """Weight per cell = (log-odds of observed rate - log-odds of predicted rate), shrunk by n/(n+k)."""
    g = pd.DataFrame({"cell": cell_key(df, key).values, "y": df.y.values, "p": sig(raw)}).groupby("cell").agg(
        n=("y", "size"), obs=("y", "mean"), pred=("p", "mean"))
    g["weight"] = (logit(g.obs) - logit(g.pred)) * g.n / (g.n + k)
    return g


def apply_weights(df, raw, weights, key):
    w = cell_key(df, key).map(weights).fillna(0).values
    return raw + w


def flags_per_1000(test, p, recall=0.9):
    e = pd.DataFrame({"y": test.y.values, "p": p, "c": test[CSN].values}).groupby("c").max().sort_values("p", ascending=False)
    caught = e.y.cumsum() / e.y.sum()
    return 1000 * (np.argmax(caught.values >= recall) + 1) / len(e)


def summarize(name, test, p, extra=None):
    s = scores(test.y.values, p, test[CSN].values)
    c, _ = calibration(pd.DataFrame({"y": test.y.values, "p": p, "c": test[CSN].values}).groupby("c").max().y.values,
                       pd.DataFrame({"y": test.y.values, "p": p, "c": test[CSN].values}).groupby("c").max().p.values)
    return {"variant": name, "AP_enc": s["AP_enc"], "AUC_enc": s["AUC_enc"], "AP_line": s["AP_line"],
            "flags_per_1000_at_90pct_recall": flags_per_1000(test, p), "ECE_enc": c["ECE"],
            "mean_pred_enc": c["mean_pred"], "actual_rate_enc": c["actual_rate"], **(extra or {})}


def cells(train, update, test, out, rules, n_sim=10, seed=7, min_n=(50, 30, 20), low_rate=0.02, high_rate=0.5):
    os.makedirs(out, exist_ok=True)
    feats = BASE_FEATS + (["RULE_PLAN_SHARE"] if rules else [])
    full = pd.concat([train, update], ignore_index=True)
    print(f"\n== Frozen base (train only) on {feats}")
    base, cats = train_lgb(train, feats)
    base.save_model(f"{out}/base_frozen.txt"); json.dump(cats, open(f"{out}/base_frozen_categories.json", "w"))
    raw = lambda m, c, df: m.predict(apply_categories(df, c, feats), raw_score=True)
    r_up, r_te = raw(base, cats, update), raw(base, cats, test)
    print("== Full model (train + update), the target to approach")
    fm, fcats = train_lgb(full, feats)
    p_full = sig(raw(fm, fcats, test))

    # ---- 1.3 isolation: weights on 5 cells change only those cells
    rng = np.random.default_rng(seed)
    key = CELL_KEYS["cpt_plan"]
    tk = cell_key(test, key)
    picks = rng.choice(tk.value_counts().index[:200], 5, replace=False)
    r_iso = apply_weights(test, r_te, pd.Series(2.0, index=picks), key)
    inside = tk.isin(picks).values
    iso = {"cells_edited": 5, "lines_in_cells": int(inside.sum()),
           "max_change_outside": float(np.abs(sig(r_iso) - sig(r_te))[~inside].max()),
           "min_change_inside": float(np.abs(sig(r_iso) - sig(r_te))[inside].min())}
    print("\n== 1.3 Isolation test:", iso, "-> PASS" if iso["max_change_outside"] == 0 and iso["min_change_inside"] > 0 else "-> FAIL")
    json.dump(iso, open(f"{out}/isolation.json", "w"), indent=1)

    # ---- 1.5 learned weights from the UPDATE window (real data, no retrain)
    rows = [summarize("frozen (no update)", test, sig(r_te)), summarize("full retrain (target)", test, p_full)]
    for kname, key in CELL_KEYS.items():
        for k in (5, 10, 20, 50, 100):
            w = learn_weights(update, r_up, key, k)
            w.to_csv(f"{out}/weights_{kname}_k{k}.csv")
            p = sig(apply_weights(test, r_te, w.weight, key))
            rows.append(summarize(f"cell weights {kname} k={k}", test, p,
                                  {"cells_with_weight": int((w.weight.abs() > 1e-9).sum()),
                                   "share_test_lines_covered": float(cell_key(test, key).isin(w.index).mean())}))
    real = pd.DataFrame(rows)
    gap = real.loc[1, "AP_enc"] - real.loc[0, "AP_enc"]
    real["gap_closed_%"] = 100 * (real.AP_enc - real.loc[0, "AP_enc"]) / gap if gap else np.nan
    real.to_csv(f"{out}/cells_real_update.csv", index=False)
    pd.set_option("display.width", 250)
    print("\n== 1.5 Learned cell weights on REAL update data (test = May-Jul)\n", real.round(4).to_string(index=False))

    # ---- 1.1c / 1.4 simulated changes: we know exactly what changed and when
    key = CELL_KEYS["cpt_plan"]
    tr_rate = pd.DataFrame({"cell": cell_key(train, key).values, "y": train.y.values}).groupby("cell").y.agg(["mean", "size"])
    n_te, n_up = cell_key(test, key).value_counts(), cell_key(update, key).value_counts()
    ok = tr_rate[(tr_rate["size"] >= min_n[0])].index.intersection(n_te[n_te >= min_n[1]].index).intersection(
        n_up[n_up >= min_n[2]].index)
    low = [c for c in ok if tr_rate.loc[c, "mean"] < low_rate]
    high = [c for c in ok if tr_rate.loc[c, "mean"] > high_rate]
    new_req = list(rng.choice(low, min(n_sim, len(low)), replace=False)) if low else []
    no_more = list(rng.choice(high, min(n_sim, len(high)), replace=False)) if high else []
    print(f"\n== Simulation: {len(new_req)} cells newly REQUIRE PA, {len(no_more)} cells NO LONGER need PA "
          f"(from the start of UPDATE)")
    up_s, te_s = update.copy(), test.copy()
    for df in (up_s, te_s):
        ck = cell_key(df, key)
        df.loc[ck.isin(new_req).values, "y"] = 1
        df.loc[ck.isin(no_more).values, "y"] = 0
    ck_te = cell_key(te_s, key)
    changed = ck_te.isin(new_req + no_more).values
    # known change on day 0: SET the cell's score to a target (weight = logit(target) - current score, per line)
    r_man = r_te.copy()
    r_man[ck_te.isin(new_req).values] = logit(0.98)
    r_man[ck_te.isin(no_more).values] = logit(0.02)
    variants = {"frozen (no update)": r_te,
                "manual cell weights: set to 0.98 / 0.02 (known change, day 0)": r_man,
                "learned cell weights k=20 (from update data)": apply_weights(
                    te_s, r_te, learn_weights(up_s, raw(base, cats, up_s), key, 20).weight, key)}
    variants["manual (day 0) + learned k=20"] = np.where(ck_te.isin(new_req + no_more).values, r_man,
                                                         variants["learned cell weights k=20 (from update data)"])
    print("   retraining full model on simulated data (comparison) ...")
    fm_s, fc_s = train_lgb(pd.concat([train, up_s], ignore_index=True), feats)
    variants["full retrain on simulated data"] = raw(fm_s, fc_s, te_s)
    srows = []
    for name, r in variants.items():
        p = sig(r)
        ch = te_s.y.values[changed]
        srows.append(summarize(name, te_s, p, {
            "changed_lines": int(changed.sum()),
            "changed_lines_correct_at_0.5": float(((p[changed] >= 0.5) == (ch == 1)).mean()) if changed.any() else np.nan,
            "newly_required_mean_p": float(p[ck_te.isin(new_req).values].mean()) if new_req else np.nan,
            "no_longer_required_mean_p": float(p[ck_te.isin(no_more).values].mean()) if no_more else np.nan,
            "max_change_unchanged_cells": float(np.abs(p - sig(r_te))[~changed].max())}))
    sim = pd.DataFrame(srows)
    sim.to_csv(f"{out}/cells_simulation.csv", index=False)
    pd.DataFrame({"cell": new_req + no_more, "change": ["newly required"] * len(new_req) + ["no longer required"] * len(no_more)}
                 ).to_csv(f"{out}/simulated_cells.csv", index=False)
    print("\n== 1.4 Simulated changes (test = May-Jul with the changes applied)\n", sim.round(4).to_string(index=False))
    print(f"\nSaved to {out}/")


# ----------------------------------------------------------------- main
def demo_data():
    tr, te, _ = pb.demo()
    d = pd.concat([tr, te], ignore_index=True)
    rng = np.random.default_rng(1)
    enc_month = pd.Series(rng.integers(1, 13, d[CSN].nunique()), index=d[CSN].unique())
    d[DATE] = pd.to_datetime(dict(year=d.YEAR, month=d[CSN].map(enc_month), day=15))
    d["Primary_DX"] = "M" + pd.Series(rng.integers(100, 400, len(d))).astype(str).values
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["months", "baselines", "cells"],
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
    ap.add_argument("--extra-feats", nargs="*", default=[],
                    help="extra claim columns for an extra model, e.g. Primary_DX QUANTITY SPECIALTY "
                         "(Primary_DX also adds DX3 = 3-char ICD category)")
    ap.add_argument("--out", default="exp_d1")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--models", nargs="*", default=None,
                    help="train only these, e.g. lgb_full (default: all lookup_* and lgb_* models)")
    ap.add_argument("--include-secondary", action="store_true",
                    help="train also on secondary/tertiary claims (own rows, own payor); TEST stays primary-only")
    ap.add_argument("--secondary-in-test", action="store_true",
                    help="with --include-secondary: keep secondary claims in TEST too (test set then differs from the base)")
    ap.add_argument("--dedup-first", action="store_true",
                    help="old repeat removal (keep first copy); default keeps a TRUE copy if any copy is TRUE")
    a = ap.parse_args()
    pb.DEDUP_KEEP_TRUE = not a.dedup_first
    pb.INCLUDE_SECONDARY = a.include_secondary

    if a.demo:
        d = demo_data()
        a.out = a.out + "_demo"
        rules = True
    else:
        import pyarrow.fs as pafs
        fs = pafs.S3FileSystem()
        raw = [c for c in BASE_FEATS if c not in (PAYOR_F, PLAN_F)] + [pb.PAYOR_NAME, pb.PLAN_NAME]
        load_extra = [("Primary_DX" if c == "DX3" else c) for c in a.extra_feats]   # DX3 is derived from Primary_DX
        cols = list(dict.fromkeys([CSN, pb.LABEL, DATE, pb.FILING] + raw + load_extra))
        print("Loading", a.train_years + a.test_years)
        d = pb.load(sorted(set(a.train_years + a.test_years)), cols, a.sample, fs)
        rules = bool(a.rules_master and a.rules_parquet)
        if rules:
            d = pb.add_rules_feature(d, a.rules_master, a.rules_parquet)

    os.makedirs(a.out, exist_ok=True)
    d = add_dx(d, a.extra_feats)
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
    print(f"TRAIN  = everything before {a.update_start}")
    print(f"UPDATE = {a.update_start} .. {a.test_start}   (used only by update methods; the 'full' models train on TRAIN+UPDATE)")

    print("\n== Split")
    train, update, test = split(d, a.update_start, a.test_start, a.test_end)
    if "IS_PRIMARY" in test and not a.secondary_in_test:       # keep the test set identical to the primary-only base
        test = test[test.IS_PRIMARY == 1].reset_index(drop=True)
        pb.log_step("TEST", "7. test: secondary claims removed (comparability)", test)
    json.dump({"update_start": a.update_start, "test_start": a.test_start, "test_end": a.test_end,
               "sample": a.sample, "rules": rules, "years": sorted(set(a.train_years + a.test_years)),
               "models": a.models, "dedup_keep_true": pb.DEDUP_KEEP_TRUE, "include_secondary": a.include_secondary, "secondary_in_test": a.secondary_in_test,
               "rows": {"train": len(train), "update": len(update), "test": len(test)}},
              open(f"{a.out}/split_info.json", "w"), indent=1)
    if min(len(train), len(update), len(test)) == 0:
        raise SystemExit("One window is empty - change --update-start / --test-start.")
    if a.mode == "cells":
        if a.demo:      # demo cells are tiny and PA is common: loosen the simulation filters
            cells(train, update, test, a.out, rules, min_n=(3, 2, 1), low_rate=0.15, high_rate=0.5)
        else:
            cells(train, update, test, a.out, rules)
    else:
        baselines(train, update, test, a.out, rules, a.extra_feats, a.models)
    f = pd.DataFrame(pb.FUNNEL)
    f.to_csv(f"{a.out}/removal_funnel.csv", index=False)
    print("\n== What was removed (also saved to removal_funnel.csv)\n", f.to_string(index=False))


if __name__ == "__main__":
    main()
