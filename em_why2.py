"""
em_why.py - WHICH feature pushes E/M false alarms above the cut-off?  (LightGBM per-feature contributions)

Uses the saved model (<run>/models/<model>.txt + <model>_categories.json). Feature rows come from test_preds.parquet
(the pipeline's own cleaned columns) plus the raw extra columns it lacks from the data file; categories are applied
with pa_base_adapter.apply_categories (same as training). Rebuilt scores are checked against the saved scores before anything is reported.
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
    for tok in ["MISSING", "UNKNOWN", "NAN", "NONE", "NULL"]:
        if tok in up:
            v = v.where(v.notna() | k.ne(""), up[tok])
            break
    return pd.Categorical(v, categories=cats), float(v.notna().mean())


def _up(s):
    return key(s).str.upper().str.replace(r"\s+", " ", regex=True).str.strip()


def _asof(s):
    return _up(s).str.replace(r"\bAS OF\b.*$", "", regex=True).str.strip()


def _words(s):
    return _asof(s).str.replace(r"[^\w\s]", " ", regex=True)


def _words1(s):
    return _words(s).str.replace(r"\s+", " ", regex=True).str.strip()


CLEANS = {"upper": _up, "asof": _asof, "punct": _words, "punct_1space": _words1}
SEPS = {"pipe_spaced": " | ", "pipe": "|", "dash": " - ", "underscore": "_", "space": " "}
DERIVE = {
    "PAYOR_CLEAN": {c: (lambda d, f=f: f(d.PAYOR)) for c, f in CLEANS.items()},
    "PAYOR_PLAN_CLEAN": {f"{c}+{sn}": (lambda d, f=f, sp=sp: f(d.PAYOR) + sp + f(d.PLAN)) for c, f in CLEANS.items() for sn, sp in SEPS.items()},
    "DX3": {"nodot3": lambda d: _up(d.Primary_DX).str.replace(".", "", regex=False).str[:3],
            "raw3": lambda d: _up(d.Primary_DX).str[:3]},
}


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

    # feature rows: start from test_preds itself (it carries the pipeline's own cleaned CPT, MOD, POS, PAYOR_CLEAN,
    # PAYOR_PLAN_CLEAN, Primary_DX, DX3 per line); only columns it lacks (SPECIALTY, DEPARTMENT_ID, VISITTYPE, NPI ...)
    # come from the data file, raw, exactly as pa_experiments loads --extra-feats
    d = t.merge(E[["k_csn", "k_cpt", "cm", "s"]], on=["k_csn", "k_cpt"])
    from_file = [f for f in feats if f not in d.columns]
    fs0, p0 = (pafs.S3FileSystem(), a.data.replace("s3://", "")) if a.data.startswith("s3://") else (pafs.LocalFileSystem(), os.path.abspath(a.data))
    have = set(ds.dataset(p0, filesystem=fs0, format="parquet").schema.names)
    direct = [f for f in from_file if f in have]
    derive = [f for f in from_file if f not in have]
    raw = [c for c in ["PAYOR", "PLAN", "Primary_DX"] if c in have] if derive else []
    print(f"features from test_preds: {[f for f in feats if f not in from_file]} | from data file: {direct} | rebuilt: {derive}")
    if direct or raw:
        f2 = read(a.data, list(dict.fromkeys(["CSN", "FINAL_CPT"] + direct + raw)))
        f2["k_csn"], f2["k_cpt"] = key(f2.CSN), key(f2.FINAL_CPT).str.upper()
        joined = False
        if "SRC_ROW" in d:                                # exact line: SRC_ROW = row number in the source year file
            rows = pd.to_numeric(d.SRC_ROW, errors="coerce")
            if "SRC_YEAR" in d:
                yr = pd.to_numeric(d.SRC_YEAR, errors="coerce")
                rows = rows.where(yr == yr.mode().iloc[0]) if yr.nunique() > 1 else rows
            for off in (0, -1):
                r = (rows + off).where((rows + off).between(0, len(f2) - 1))
                ok = r.notna()
                m = pd.Series(False, index=d.index)
                m[ok] = f2.k_csn.values[r[ok].astype(int)] == d.k_csn[ok].values
                if m.mean() > 0.99:
                    for c in direct + raw:
                        d[c] = f2[c].values[r.fillna(0).astype(int)]
                    print(f"joined data-file columns by SRC_ROW (offset {off}): {m.mean():.1%} of lines match on CSN")
                    joined = True
                    break
        if not joined:                                    # fallback: first line of the same CSN + CPT
            print("joined data-file columns by CSN + CPT (first line) - SRC_ROW not usable")
            g = f2[f2.k_csn.isin(set(E.k_csn))].drop(columns=["CSN", "FINAL_CPT"]).drop_duplicates(["k_csn", "k_cpt"])
            d = d.drop(columns=[c for c in direct + raw if c in d]).merge(g, on=["k_csn", "k_cpt"], how="left")
    for f in derive:                                      # last resort: try known recipes, keep the best match
        if f not in DERIVE or f not in cats:
            raise SystemExit(f"model feature {f} has no source - send: grep -n '{f}' pa_base_adapter.py")
        best = max(((name, fn(d)) for name, fn in DERIVE[f].items()), key=lambda r: to_model_cats(r[1], cats[f])[1])
        d[f] = best[1]
        print(f"built {f} with recipe '{best[0]}'")
    known = {f: (to_model_cats(d[f], cats[f])[1] if f in cats else float(pd.to_numeric(d[f], errors='coerce').notna().mean())) for f in feats}
    try:                                                  # the pipeline's own category mapping = exactly what training used
        import pa_base_adapter as pb
        X = pb.apply_categories(d, cats, feats)
        print("categories applied with pa_base_adapter.apply_categories")
    except Exception as ex:
        print(f"pa_base_adapter.apply_categories not usable ({type(ex).__name__}: {ex}) - using string matching")
        X = pd.DataFrame(index=d.index)
        for f in feats:
            X[f] = to_model_cats(d[f], cats[f])[0] if f in cats else pd.to_numeric(d[f], errors="coerce")
    X = X[feats]
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
