"""
em_why.py - WHICH feature pushes E/M false alarms above the cut-off?  (LightGBM per-feature contributions)

Uses the saved model (<run>/models/<model>.txt + <model>_categories.json), rebuilds the feature rows of the test E/M
lines from the data file, and checks the rebuilt scores against <run>/test_preds.parquet before trusting anything.
For each E/M service the line with the highest score is used (service score = max over lines, as everywhere else).

  0_score_check        rebuilt score vs saved score (must match, else stop and fix the feature rebuild)
                       + per feature: share of rows whose value is a known category of the model
  1_mean_contrib       mean contribution (log-odds) of each feature for E/M TP / FP / FN / TN
  2_top_driver         for each E/M FP: the feature with the largest positive contribution -> share of FP per feature
  3_<feature>          for the top drivers (DEPARTMENT_ID, Primary_DX, SPECIALTY, PAYOR_PLAN_CLEAN): FP count and mean
                       contribution per value (top 25). Provider NPI: concentration only (top-10 / top-50 share of FP);
                       the per-NPI list is written to <out>/npi_fp.csv on EFS and not printed.
Counts and contributions only; no patient IDs.

  nohup python -u em_why.py --run exp_feat_core_s42 --out em_why_core > run_em_why.log 2>&1 &
"""
import argparse
import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.fs as pafs

DATA = "s3://cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v8_draft/ai_prior_auth_09282026_v3_2026_std.parquet"
NPI = "SERVICEPROVIDERNPI"
DETAIL = ["DEPARTMENT_ID", "Primary_DX", "SPECIALTY", "PAYOR_PLAN_CLEAN", "VISITTYPE"]


def key(s):
    return s.astype(object).where(s.notna(), None).map(lambda v: "" if v is None else str(v)) \
        .str.strip().str.replace(r"^(\d+)\.0$", r"\1", regex=True)          # 1001.0 -> 1001, but Z00.0 stays


def is_em(cpt):
    n = pd.to_numeric(cpt.where(cpt.str.fullmatch(r"\d{5}")), errors="coerce")
    return n.between(99202, 99499)


def read(path, cols, csns=None):
    fs, p = (pafs.S3FileSystem(), path.replace("s3://", "")) if path.startswith("s3://") else (pafs.LocalFileSystem(), os.path.abspath(path))
    d = ds.dataset(p, filesystem=fs, format="parquet")
    miss = [c for c in cols if c not in d.schema.names]
    if miss:
        raise SystemExit(f"model features not in the data file: {miss} - tell me how pa_experiments builds them")
    return d.to_table(columns=cols).to_pandas()


def to_model_cats(x, cats):
    """map raw values onto the model's category list by string (then upper-case) match; unknown -> NaN"""
    look = {}
    for c, k in zip(cats, key(pd.Series(cats, dtype=object))):
        look.setdefault(k, c)
    up = {k.upper(): v for k, v in look.items()}
    k = key(x)
    v = k.map(look)
    v = v.where(v.notna(), k.str.upper().map(up))
    return pd.Categorical(v, categories=cats), float(v.notna().mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="lgb_ctx_full")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--recall", type=float, default=0.85)
    ap.add_argument("--tn-sample", type=int, default=50000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)

    bst = lgb.Booster(model_file=f"{a.run}/models/{a.model}.txt")
    feats = bst.feature_name()
    cj = f"{a.run}/models/{a.model}_categories.json"
    cats = json.load(open(cj)) if os.path.exists(cj) else {}
    print(f"model features ({len(feats)}): {feats}\ncategory lists for: {sorted(cats)}")

    # test services, cut-off, E/M classes (same as em_check / referral_check)
    t = pd.read_parquet(f"{a.run}/test_preds.parquet")
    t = t.loc[:, ~t.columns.duplicated()]
    pcol = [c for c in t.columns if c.startswith("p_") and a.model in c] or [c for c in t.columns if c.startswith("p_")]
    pcol = pcol[0]
    t["k_csn"], t["k_cpt"] = key(t.CSN), key(t.FINAL_CPT).str.upper()
    S = t.groupby(["k_csn", "k_cpt"]).agg(y=("y", "max"), s=(pcol, "max")).reset_index()
    S["y"] = S.y.astype(bool)
    cut = float(np.quantile(S.s[S.y], 1 - a.recall))
    S["cm"] = np.select([S.y & (S.s >= cut), ~S.y & (S.s >= cut), S.y], ["TP", "FP", "FN"], "TN")
    E = S[is_em(S.k_cpt)]
    tn = E[E.cm == "TN"]
    E = pd.concat([E[E.cm != "TN"], tn.sample(min(a.tn_sample, len(tn)), random_state=0)])
    print(f"score column {pcol} | cut-off {cut:.6f} | E/M services used: " + ", ".join(f"{k} {v:,}" for k, v in E.cm.value_counts().items()))

    # rebuild feature rows for those services' lines
    d = read(a.data, list(dict.fromkeys(["CSN", "FINAL_CPT"] + feats)))
    d["k_csn"], d["k_cpt"] = key(d.CSN), key(d.FINAL_CPT).str.upper()
    d = d.merge(E[["k_csn", "k_cpt", "cm", "s"]], on=["k_csn", "k_cpt"])
    X = pd.DataFrame(index=d.index)
    known = {}
    for f in feats:
        if f in cats:
            X[f], known[f] = to_model_cats(d[f], cats[f])
        else:
            X[f] = pd.to_numeric(d[f], errors="coerce")
            known[f] = float(X[f].notna().mean())
    d["s_rebuilt"] = bst.predict(X)
    C = pd.DataFrame(bst.predict(X, pred_contrib=True), columns=feats + ["base"], index=d.index)

    # keep the top-scoring line per service
    top = d.sort_values("s_rebuilt", ascending=False).drop_duplicates(["k_csn", "k_cpt"]).index
    d, C = d.loc[top], C.loc[top]
    diff = (d.s_rebuilt - d.s).abs()
    chk = pd.DataFrame({"value": [len(d), float((diff < 1e-4).mean()), float(diff.median()), float(np.corrcoef(d.s, d.s_rebuilt)[0, 1])]},
                       index=["services", "share_within_1e-4", "median_abs_diff", "corr"])
    kn = pd.DataFrame({"value": known}).rename(index=lambda f: f"known_category:{f}")
    out = {"0_score_check": pd.concat([chk, kn]).round(4)}
    if chk.value["share_within_1e-4"] < 0.95:
        print(out["0_score_check"].to_string())
        print("\nSTOP: rebuilt scores do not match test_preds - the feature rebuild differs from training. Send me this output.")
        out["0_score_check"].to_csv(f"{a.out}/0_score_check.csv")
        os._exit(1)

    C["cm"] = d.cm.values
    out["1_mean_contrib"] = C.groupby("cm")[feats + ["base"]].mean().T.reindex(columns=["TP", "FP", "FN", "TN"]).round(3)
    fp = C[C.cm == "FP"]
    drv = fp[feats].idxmax(axis=1)
    out["2_top_driver"] = pd.DataFrame({"FP": drv.value_counts(), "share": drv.value_counts(normalize=True).round(3),
                                        "mean_contrib_when_top": fp[feats].max(axis=1).groupby(drv).mean().round(3)})
    D = d.assign(**{f"c_{f}": C[f] for f in feats})
    for f in [x for x in DETAIL if x in feats]:
        g = D.groupby(key(D[f]).replace("", "(blank)"))
        tab = pd.DataFrame({"FP": g.cm.apply(lambda s: (s == "FP").sum()), "TP": g.cm.apply(lambda s: (s == "TP").sum()),
                            f"mean_contrib_FP": D[D.cm == "FP"].groupby(key(D[D.cm == "FP"][f]).replace("", "(blank)"))[f"c_{f}"].mean().round(3)})
        out[f"3_{f}"] = tab.sort_values("FP", ascending=False).head(25)
    if NPI in feats:
        n = D[D.cm == "FP"].groupby(key(D[D.cm == "FP"][NPI])).size().sort_values(ascending=False)
        n.rename("FP").to_csv(f"{a.out}/npi_fp.csv")                                   # stays on EFS
        out["3_NPI_concentration"] = pd.DataFrame({"value": [len(n), n.head(10).sum() / n.sum(), n.head(50).sum() / n.sum(),
                                                             float(D.loc[D.cm == 'FP', f'c_{NPI}'].mean())]},
                                                  index=["NPIs with an E/M FP", "top-10 share", "top-50 share", "mean NPI contrib on FP"]).round(3)
    for k, v in out.items():
        v.to_csv(f"{a.out}/{k}.csv")
        print(f"\n== {k}\n{v.to_string()}")
    print(f"\noutputs -> {a.out}/  (contributions are log-odds; positive = pushes towards PA)")
    os._exit(0)


if __name__ == "__main__":
    main()


nohup python -u em_why.py --run exp_feat_core_s42 --out em_why_core > run_em_why.log 2>&1 &
tail -f run_em_why.log
