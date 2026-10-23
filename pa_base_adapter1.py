"""
pa_base_adapter.py - PA model v0: frozen BASE model + ADAPTER that adds new feature(s).

  BASE     LightGBM on the initial features. Saved and never changed again.
  ADAPTER  extra trees trained on top of the base score (init_score), using base + new features.
           with adapter:    p = sigmoid(base_raw + adapter_raw)
           without adapter: p = base model alone (identical to before - checked at the end)
  FULL     (optional --full) one model trained from scratch on all features, to compare.

New feature options
  --new-feats COL [COL ...]         any column already in the claims (e.g. LINE_OF_BUSINESS)
  --rules-master X.xlsx --rules-parquet all_rules.parquet
                                    builds RULE_PLAN_SHARE = share of the rules plans of this claim's
                                    payor code (XREF) that list this CPT. 1 = every plan lists it,
                                    0 = none, empty = payor not linked to rules. PROVISIONAL: uses
                                    the name-based master sheet (payor level), not the SME mapping.

Run (SageMaker terminal):
  pip install lightgbm            # if missing
  python pa_base_adapter.py --train-years 2024 2025 --test-year 2026 --sample 0.3 \
      --rules-master <...>/payor_master_review.xlsx --rules-parquet rules_analysis/all_rules.parquet --full
  python pa_base_adapter.py --demo   # synthetic data, checks the mechanics

Outputs (--out, default pa_model_v0/): base_model.txt, adapter_model.txt, full_model.txt,
  categories.json, metrics.csv, metrics_by_segment.csv, feature_importance.csv, run_info.json
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

# ----------------------------------------------------------------- column names (edit if yours differ)
BASE_PATH = ("s3://cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v4/"
             "ai_prior_auth_09282026_v3_{year}_std.parquet")
CSN, LABEL, DATE, FILING = "CSN", "GROUP_LABEL", "SERVICEDATE", "FILING_ORDER_C"
CPT, MOD, POS = "CPT", "FINAL_MODIFIERS", "PLACE_OF_SERVICE_CODE"
PAYOR_ID, PLAN_ID = "BUCKET_PAYOR_ID", "BUCKET_PLAN_ID"
PAYOR_NAME, PLAN_NAME = "PAYOR", "PLAN"          # names used in the master sheet
SEGMENT = "SEGMENT"                               # filled from --segments file if given
# BUCKET_PAYOR_ID / BUCKET_PLAN_ID are NOT payor/plan keys (one ID holds hundreds of payor names),
# so the payor/plan features are the NAMES, cleaned (see clean_names).
PAYOR_F, PLAN_F = "PAYOR_CLEAN", "PAYOR_PLAN_CLEAN"
BASE_FEATS = [CPT, MOD, POS, PAYOR_F, PLAN_F]

PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              cat_smooth=20, cat_l2=10, max_cat_to_onehot=8, verbose=-1, seed=42)
ADAPTER_PARAMS = dict(PARAMS, learning_rate=0.03, num_leaves=31, min_data_in_leaf=200)


# ----------------------------------------------------------------- data
def to_label(s):
    s = s.astype(str).str.strip().str.upper()
    return s.map({"TRUE": 1, "1": 1, "1.0": 1, "FALSE": 0, "0": 0, "0.0": 0})   # anything else -> NaN (dropped)


def load(years, cols, sample, fs=None):
    parts = []
    for y in years:
        t0 = time.time()
        d = pd.read_parquet(BASE_PATH.format(year=y), columns=cols, filesystem=fs)
        d = prep(d, sample)
        d["YEAR"] = y
        parts.append(d)
        print(f"  {y}: {len(d):,} lines, {d[CSN].nunique():,} encounters, PA line rate {d.y.mean():.3%} "
              f"({time.time()-t0:.0f}s)")
    return pd.concat(parts, ignore_index=True)


def prep(d, sample):
    d = d.copy()
    d["y"] = to_label(d[LABEL])
    d = d[d.y.notna()]                                           # drop AMBIGUOUS / missing labels
    if FILING in d:
        d = d[pd.to_numeric(d[FILING], errors="coerce") == 1]    # primary claims only
    d = d.drop_duplicates([c for c in [CSN, CPT, MOD, DATE] if c in d])   # repeated lines
    if sample < 1:                                               # sample whole encounters
        keep = d[CSN].drop_duplicates().sample(frac=sample, random_state=42)
        d = d[d[CSN].isin(keep)]
    d["y"] = d.y.astype(int)
    return clean_names(d)


def clean_names(d):
    """Upper-case, squeeze spaces, drop date suffixes like 'AS OF 01/01/2026' so renames stay one plan."""
    def c(s):
        s = s.astype(str).str.upper().str.replace(r"\bAS OF\b.*$", "", regex=True)
        return s.str.replace(r"\s+", " ", regex=True).str.strip()
    d[PAYOR_F] = c(d[PAYOR_NAME])
    d[PLAN_F] = d[PAYOR_F] + " | " + c(d[PLAN_NAME])
    return d


# ----------------------------------------------------------------- provisional rules feature
def add_rules_feature(d, master_xlsx, rules_parquet):
    m = pd.read_excel(master_xlsx, usecols=[PAYOR_NAME, PLAN_NAME, "rules_payor"])
    m = m.astype(str).apply(lambda s: s.str.strip().str.upper()).drop_duplicates([PAYOR_NAME, PLAN_NAME])
    r = pd.read_parquet(rules_parquet, columns=["file_profile", "XREF_ID", "MEDICAL_CODES"])
    r["XREF_ID"] = r.XREF_ID.astype(str).str.strip().str.upper()
    r["code"] = r.MEDICAL_CODES.astype(str).str.strip().str.upper()
    n_plans = r.groupby("XREF_ID").file_profile.nunique().rename("n_plans")
    listed = (r.drop_duplicates(["XREF_ID", "file_profile", "code"])
              .groupby(["XREF_ID", "code"]).file_profile.nunique().rename("plans_listing").reset_index())

    key = d[[PAYOR_NAME, PLAN_NAME, CPT]].astype(str).apply(lambda s: s.str.strip().str.upper())
    key = key.merge(m, on=[PAYOR_NAME, PLAN_NAME], how="left")
    key = key.merge(n_plans, left_on="rules_payor", right_index=True, how="left")
    key = key.merge(listed, left_on=["rules_payor", CPT], right_on=["XREF_ID", "code"], how="left")
    share = key.plans_listing.fillna(0) / key.n_plans                # NaN when payor not linked
    d["RULE_PLAN_SHARE"] = share.values
    linked = d.RULE_PLAN_SHARE.notna()
    print(f"  rules feature: {linked.mean():.1%} of lines linked to a rules payor; "
          f"PA rate when listed by all plans {d.loc[d.RULE_PLAN_SHARE == 1, 'y'].mean():.2%}, "
          f"by none {d.loc[d.RULE_PLAN_SHARE == 0, 'y'].mean():.2%}")
    return d


# ----------------------------------------------------------------- categories (fixed from training data)
def fit_categories(train, cols, min_count=20):
    cats = {}
    for c in cols:
        if train[c].dtype.kind in "biuf" and c not in BASE_FEATS:
            continue                                              # numeric feature
        vc = train[c].astype(str).value_counts()
        cats[c] = sorted(vc[vc >= min_count].index)               # rare values -> NaN (treated as missing)
    return cats


def apply_categories(d, cats, cols):
    X = pd.DataFrame(index=d.index)
    for c in cols:
        X[c] = pd.Categorical(d[c].astype(str), categories=cats[c]) if c in cats else pd.to_numeric(d[c], errors="coerce")
    return X


# ----------------------------------------------------------------- metrics
def scores(y, p, csn):
    enc = pd.DataFrame({"y": y, "p": p, "csn": csn}).groupby("csn").max()   # encounter = any line
    return {"lines": len(y), "pa_line_rate": y.mean(),
            "AP_line": average_precision_score(y, p), "AUC_line": roc_auc_score(y, p),
            "AP_lift_line": average_precision_score(y, p) / y.mean(),
            "encounters": len(enc), "AP_enc": average_precision_score(enc.y, enc.p),
            "AUC_enc": roc_auc_score(enc.y, enc.p)}


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# ----------------------------------------------------------------- main
def run(train, test, new_feats, out, full, rounds):
    os.makedirs(out, exist_ok=True)
    all_feats = BASE_FEATS + new_feats
    cats = fit_categories(train, all_feats)
    json.dump(cats, open(f"{out}/categories.json", "w"))

    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)     # validation split by encounter
    tr_i, va_i = next(gss.split(train, groups=train[CSN]))
    tr, va = train.iloc[tr_i], train.iloc[va_i]
    Xtr, Xva, Xte = (apply_categories(x, cats, all_feats) for x in (tr, va, test))
    stop = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)]

    # 1) BASE
    print("\n== training BASE on", BASE_FEATS)
    base = lgb.train(PARAMS, lgb.Dataset(Xtr[BASE_FEATS], tr.y), rounds,
                     valid_sets=[lgb.Dataset(Xva[BASE_FEATS], va.y)], callbacks=stop)
    base.save_model(f"{out}/base_model.txt")
    base_raw = {k: base.predict(X[BASE_FEATS], raw_score=True) for k, X in (("tr", Xtr), ("va", Xva), ("te", Xte))}

    # 2) ADAPTER on top of the frozen base score
    print("\n== training ADAPTER with", new_feats, "on top of BASE")
    adapter = lgb.train(ADAPTER_PARAMS, lgb.Dataset(Xtr[all_feats], tr.y, init_score=base_raw["tr"]), rounds,
                        valid_sets=[lgb.Dataset(Xva[all_feats], va.y, init_score=base_raw["va"])], callbacks=stop)
    adapter.save_model(f"{out}/adapter_model.txt")

    preds = {"base": sigmoid(base_raw["te"]),
             "base+adapter": sigmoid(base_raw["te"] + adapter.predict(Xte[all_feats], raw_score=True))}
    models = {"base": base, "adapter": adapter}

    # 3) FULL retrain for comparison
    if full:
        print("\n== training FULL model from scratch on", all_feats)
        fm = lgb.train(PARAMS, lgb.Dataset(Xtr[all_feats], tr.y), rounds,
                       valid_sets=[lgb.Dataset(Xva[all_feats], va.y)], callbacks=stop)
        fm.save_model(f"{out}/full_model.txt")
        preds["full_retrain"] = fm.predict(Xte[all_feats])
        models["full"] = fm

    # base really unchanged? reload from disk and compare
    same = np.allclose(lgb.Booster(model_file=f"{out}/base_model.txt").predict(Xte[BASE_FEATS]), preds["base"])
    print(f"\nBase model unchanged after adapter training: {same}")

    # metrics overall + per segment
    met = pd.DataFrame({k: scores(test.y.values, p, test[CSN].values) for k, p in preds.items()}).T
    met.to_csv(f"{out}/metrics.csv")
    print("\n== TEST metrics (", sorted(test.YEAR.unique()), ")\n", met.round(4).to_string())
    if SEGMENT in test:
        rows = []
        for seg, g in test.groupby(SEGMENT):
            if g.y.nunique() < 2 or g.y.sum() < 50:
                continue
            for k, p in preds.items():
                s = scores(g.y.values, p[test.index.get_indexer(g.index)], g[CSN].values)
                rows.append({"segment": seg, "model": k, **s})
        seg_met = pd.DataFrame(rows)
        seg_met.to_csv(f"{out}/metrics_by_segment.csv", index=False)
        print("\n== AP (encounter) by segment\n",
              seg_met.pivot(index="segment", columns="model", values="AP_enc").round(3).to_string())

    imp = pd.concat({k: pd.Series(m.feature_importance("gain"), index=m.feature_name())
                     for k, m in models.items()}, axis=1)
    (imp / imp.sum()).round(3).to_csv(f"{out}/feature_importance.csv")
    print("\n== share of gain by feature\n", (imp / imp.sum()).round(3).to_string())
    json.dump({"base_feats": BASE_FEATS, "new_feats": new_feats, "base_trees": base.num_trees(),
               "adapter_trees": adapter.num_trees(), "train_lines": len(train), "test_lines": len(test),
               "train_years": sorted(map(int, train.YEAR.unique())), "test_years": sorted(map(int, test.YEAR.unique()))},
              open(f"{out}/run_info.json", "w"), indent=1)
    print(f"\nSaved to {out}/")


def demo():
    """Synthetic claims: PA depends on CPT x payor, plus a hidden 'rule' signal only the new feature sees."""
    rng = np.random.default_rng(0)
    n_enc = 60000
    enc = pd.DataFrame({CSN: np.arange(n_enc), PAYOR_NAME: rng.integers(0, 30, n_enc).astype(str),
                        PLAN_NAME: rng.integers(0, 80, n_enc).astype(str), POS: rng.choice(["11", "22", "19"], n_enc),
                        "YEAR": rng.choice([2024, 2025, 2026], n_enc, p=[.4, .35, .25]),
                        SEGMENT: rng.choice(["Commercial", "Medicaid", "MA"], n_enc)})
    d = enc.loc[enc.index.repeat(rng.integers(1, 5, n_enc))].reset_index(drop=True)
    d[CPT] = rng.integers(0, 400, len(d)).astype(str)
    d[MOD] = rng.choice(["", "26", "TC", "59"], len(d))
    listed = rng.random((30, 400)) < 0.15                               # payor rule lists
    d["RULE_PLAN_SHARE"] = listed[d[PAYOR_NAME].astype(int), d[CPT].astype(int)].astype(float)
    cpt_eff = rng.normal(-3.5, 1, 400)
    z = cpt_eff[d[CPT].astype(int)] + 2.5 * d.RULE_PLAN_SHARE + (d[POS] == "22") * 0.5
    d["y"] = (rng.random(len(d)) < sigmoid(z)).astype(int)
    d["y"] = d.groupby(CSN).y.transform("max")                         # label per encounter, like ours
    d = clean_names(d)
    return d[d.YEAR < 2026], d[d.YEAR == 2026].reset_index(drop=True), ["RULE_PLAN_SHARE"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-years", nargs="+", type=int, default=[2024, 2025])
    ap.add_argument("--test-year", type=int, default=2026)
    ap.add_argument("--sample", type=float, default=0.3, help="share of encounters kept (memory)")
    ap.add_argument("--new-feats", nargs="*", default=[])
    ap.add_argument("--rules-master"); ap.add_argument("--rules-parquet")
    ap.add_argument("--segments", help="optional csv/xlsx with PAYOR, PLAN, segment (by NAME, not bucket ids)")
    ap.add_argument("--full", action="store_true", help="also train a full model from scratch")
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--out", default="pa_model_v0")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()

    if a.demo:
        train, test, new = demo()
        return run(train, test, new, a.out + "_demo", True, a.rounds)

    import pyarrow.fs as pafs
    fs = pafs.S3FileSystem()
    raw = [c for c in BASE_FEATS if c not in (PAYOR_F, PLAN_F)] + [PAYOR_NAME, PLAN_NAME]
    cols = list(dict.fromkeys([CSN, LABEL, DATE, FILING] + raw + a.new_feats))
    print("Loading train", a.train_years)
    train = load(a.train_years, cols, a.sample, fs)
    print("Loading test", a.test_year)
    test = load([a.test_year], cols, a.sample, fs)

    new = list(a.new_feats)
    if a.rules_master and a.rules_parquet:
        train = add_rules_feature(train, a.rules_master, a.rules_parquet)
        test = add_rules_feature(test, a.rules_master, a.rules_parquet)
        new.append("RULE_PLAN_SHARE")
    if not new:
        raise SystemExit("Give --new-feats and/or --rules-master + --rules-parquet for the adapter.")

    if a.segments:
        s = pd.read_excel(a.segments) if a.segments.endswith("xlsx") else pd.read_csv(a.segments)
        s = s.rename(columns={c: SEGMENT for c in s.columns if c.lower() in ("segment", "final_segment")})
        s = clean_names(s[[PAYOR_NAME, PLAN_NAME, SEGMENT]].copy())[[PLAN_F, SEGMENT]].drop_duplicates(PLAN_F)
        train = train.merge(s, on=PLAN_F, how="left")
        test = test.merge(s, on=PLAN_F, how="left")

    run(train, test.reset_index(drop=True), new, a.out, a.full, a.rounds)


if __name__ == "__main__":
    main()
