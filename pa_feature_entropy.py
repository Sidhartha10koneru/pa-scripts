"""
pa_feature_entropy.py - EDA before modelling: which inputs carry information about PA?

Runs on the TRAINING window only (default Jan 2025 - Jan 2026), never the test months.
All scores are out-of-fold (visits split in 2 halves: learn group PA rates on one half, measure on the other),
so detailed groupings don't look good just because tiny groups are "pure" by chance.

Tables
  A. single features          - how much PA uncertainty each input removes on its own (feature ranking)
  B. cell definitions         - which combination of fields separates PA best (how detailed a cell should be)
  C. add-on to the current cell - does a NEW input (age, units, specialty, diagnosis, visit context...) add information
                                  BEYOND the current 5 features? (rates smoothed towards the current cell's rate)

Metrics
  info_gain_%  share of the label's uncertainty (entropy) removed, out-of-fold. Higher = more information.
  AP_line      ranking quality of the group PA rate as a score (same metric as the models).
  groups       number of distinct groups.

Run:
  python pa_feature_entropy.py --out entropy_train 2>&1 | tee run_entropy_train.log
  python pa_feature_entropy.py --demo                 # synthetic check
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

import pa_base_adapter as pb
import pa_experiments as pe

C, M, P, PY, PL = "FINAL_CPT", "FINAL_MODIFIERS", "PLACE_OF_SERVICE_CODE", "PAYOR_CLEAN", "PAYOR_PLAN_CLEAN"
CELL = [C, M, P, PL]
# candidate new columns (used only if present in the file)
CANDIDATES = {"BIRTH_DATE": "AGE_BAND", "QUANTITY": "UNITS_BAND", "SPECIALTY": "SPECIALTY", "Primary_DX": "DX",
              "PRODUCT_NAME": "PRODUCT_NAME", "VISITTYPE": "VISITTYPE", "LINE_OF_BUSINESS": "LINE_OF_BUSINESS",
              "PATIENT_SEX": "SEX", "SEX": "SEX", "DEPARTMENT_NAME": "DEPARTMENT"}


def H(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))


def key_of(t, cols):
    k = t[cols[0]].astype(str)
    for c in cols[1:]:
        k = k + "|" + t[c].astype(str)
    return k


def oof_rate(t, fold, cols, prior, m=20):
    """out-of-fold smoothed PA rate of each line's group; prior = per-line fallback probability"""
    key = key_of(t, cols)
    p = np.empty(len(t))
    for f in (0, 1):
        tr, te = fold != f, fold == f
        g = pd.DataFrame({"k": key[tr].values, "y": t.y.values[tr], "pr": prior[tr]}).groupby("k").agg(
            s=("y", "sum"), n=("y", "size"), pr=("pr", "mean"))
        k_te = key[te]
        s, n, pr_g = k_te.map(g.s), k_te.map(g.n), k_te.map(g.pr)
        pri = prior[te]
        p[te] = np.where(n.isna(), pri, (s.fillna(0) + m * pri) / (n.fillna(0) + m))
    return np.clip(p, 1e-6, 1 - 1e-6), key.nunique()


def metrics(y, p, H0, groups):
    ce = float(-(y * np.log2(p) + (1 - y) * np.log2(1 - p)).mean())
    return {"groups": groups, "entropy_left_bits": ce, "info_gain_%": 100 * (1 - ce / H0),
            "AP_line": average_precision_score(y, p)}


def visit_context(t):
    """what else is in the visit (independent of this line): families present on OTHER lines"""
    fam = t[C].map(cpt_family)
    t["FAMILY"] = fam
    flags = {}
    for name, fams in {"imaging": ["imaging"], "drug": ["HCPCS J drugs", "infusion/chemo admin"],
                       "surgery": ["surgery"], "rehab": ["PT/OT rehab"], "contrast": ["HCPCS A supplies/contrast", "HCPCS Q"]}.items():
        hit = fam.isin(fams).astype(int)
        flags[name] = (hit.groupby(t["CSN"]).transform("sum") - hit) > 0          # another line in the visit
    ctx = np.full(len(t), "", dtype=object)
    for name, f in flags.items():
        ctx = np.where(f.values, ctx + name + ",", ctx)
    t["VISIT_CONTEXT"] = pd.Series(ctx, index=t.index).replace("", "none")
    n = t.groupby("CSN")[C].transform("size")
    t["VISIT_LINES"] = pd.cut(n, [0, 1, 2, 4, 8, 1e9], labels=["1", "2", "3-4", "5-8", "9+"]).astype(str)
    return t


def cpt_family(c):
    from pa_diagnostics import cpt_family as f
    return f(c)


def prepare(t):
    for c in [C, M, P, PY, PL]:
        t[c] = t[c].astype(str)
    t["RULES"] = t["RULE_PLAN_SHARE"].round(2).astype(str) if "RULE_PLAN_SHARE" in t else "na"
    if "BIRTH_DATE" in t:
        age = (t.SVC_DATE - pd.to_datetime(t.BIRTH_DATE, errors="coerce")).dt.days / 365.25
        t["AGE_BAND"] = pd.cut(age, [-1, 1, 6, 18, 21, 40, 65, 75, 200],
                               labels=["<1", "1-5", "6-17", "18-20", "21-39", "40-64", "65-74", "75+"]).astype(str)
    if "QUANTITY" in t:
        q = pd.to_numeric(t.QUANTITY, errors="coerce")
        t["UNITS_BAND"] = pd.cut(q, [-1e9, 0, 1, 2, 4, 10, 50, 1e9], labels=["<=0", "1", "2", "3-4", "5-10", "11-50", "50+"]).astype(str)
    if "Primary_DX" in t:
        t["DX"] = t.Primary_DX.astype(str).str.upper().str.replace(".", "", regex=False).str[:3]     # ICD-10 category
    for src, dst in CANDIDATES.items():
        if src in t and dst not in t:
            t[dst] = t[src].astype(str)
    return visit_context(t)


def run(t, out):
    os.makedirs(out, exist_ok=True)
    pd.set_option("display.width", 250)
    y = t.y.values.astype(float)
    base = y.mean()
    H0 = H(base)
    fold = (pd.util.hash_pandas_object(t["CSN"], index=False) % 2).values
    flat = np.full(len(t), base)
    print(f"\nTRAIN window: {len(t):,} lines, {t.CSN.nunique():,} visits, PA rate {base:.3%}, entropy {H0:.4f} bits")

    def table(title, d, prior=None, name=None):
        rows = {}
        for k, cols in d.items():
            if not all(c in t for c in cols):
                continue
            p, g = oof_rate(t, fold, cols, flat if prior is None else prior)
            rows[k] = metrics(y, p, H0, g)
        r = pd.DataFrame(rows).T.sort_values("info_gain_%", ascending=False)
        print(f"\n== {title}\n", r.round(4).to_string())
        r.to_csv(f"{out}/{name}.csv")
        return r

    singles = {"CPT": [C], "modifier": [M], "POS": [P], "payor": [PY], "payor+plan": [PL], "rules value": ["RULES"],
               "CPT family": ["FAMILY"], "visit context": ["VISIT_CONTEXT"], "visit lines": ["VISIT_LINES"],
               **{v: [v] for v in set(CANDIDATES.values()) | {"AGE_BAND", "UNITS_BAND", "DX"} if v in t}}
    table("A. SINGLE FEATURES (feature ranking)", singles, name="A_single_features")

    combos = {"CPT+payor": [C, PY], "CPT+payor+plan": [C, PL], "CPT+POS": [C, P], "CPT+mod": [C, M],
              "CPT+payor+plan+POS": [C, PL, P], "CPT+payor+plan+mod": [C, PL, M],
              "PRD cell: CPT+mod+POS+payor+plan": CELL, "CPT+rules": [C, "RULES"], "CPT+payor+rules": [C, PY, "RULES"]}
    table("B. CELL DEFINITIONS", combos, name="B_cell_definitions")

    # C: does a new input add information beyond the current cell? prior = out-of-fold rate of the PRD cell
    cell_p, _ = oof_rate(t, fold, CELL, flat)
    now = metrics(y, cell_p, H0, None)
    adds = {f"+ {v}": CELL + [v] for v in ["AGE_BAND", "UNITS_BAND", "SPECIALTY", "DX", "PRODUCT_NAME", "VISITTYPE",
                                           "LINE_OF_BUSINESS", "SEX", "DEPARTMENT", "VISIT_CONTEXT", "VISIT_LINES", "RULES"] if v in t}
    r = table("C. ADD-ON TO THE CURRENT CELL (smoothed towards the cell's own rate)", adds, prior=cell_p, name="C_addons")
    r["extra_gain_vs_cell_pts"] = r["info_gain_%"] - now["info_gain_%"]
    r["extra_AP_vs_cell"] = r["AP_line"] - now["AP_line"]
    print(f"\n   current cell alone: info_gain {now['info_gain_%']:.2f}% | AP_line {now['AP_line']:.4f}")
    print("   extra information each new input adds beyond the current cell:\n",
          r[["extra_gain_vs_cell_pts", "extra_AP_vs_cell"]].round(4).to_string())
    r.to_csv(f"{out}/C_addons.csv")
    print(f"\nSaved to {out}/")


def demo():
    rng = np.random.default_rng(0)
    n = 80000
    csn = rng.integers(0, 30000, n)
    t = pd.DataFrame({"CSN": csn, C: rng.choice(["99214", "70553", "97110", "J1745", "3074F", "80053"], n),
                      M: rng.choice(["", "GP", "26"], n), P: rng.choice(["11", "22"], n),
                      PY: rng.choice(["UHC", "AETNA"], n)})
    t[PL] = t[PY] + " | " + rng.choice(["PPO", "MEDICAID"], n)
    t["SVC_DATE"] = pd.Timestamp("2025-06-01")
    t["BIRTH_DATE"] = pd.Timestamp("2025-06-01") - pd.to_timedelta(rng.integers(0, 90 * 365, n), unit="D")
    t["QUANTITY"] = rng.choice([1, 1, 1, 2, 10], n)
    age = (t.SVC_DATE - t.BIRTH_DATE).dt.days / 365
    p = t[C].map({"70553": .4, "97110": .1, "J1745": .3}).fillna(.005) * np.where(age < 21, 3, 1)
    t["y"] = (rng.random(n) < np.clip(p, 0, .95)).astype(int)
    t["RULE_PLAN_SHARE"] = rng.choice([0, .5, 1], n)
    return t


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--sample", type=float, default=1.0)
    a.add_argument("--update-start", default="2026-02-01")
    a.add_argument("--rules-master", default="/home/sagemaker-user/user-default-efs/ksidhar/v3data_eda_analysis/Prd_rules_analysis/payor_master_review.xlsx")
    a.add_argument("--rules-parquet", default="rules_analysis/all_rules.parquet")
    a.add_argument("--out", default="entropy_train")
    a.add_argument("--demo", action="store_true")
    o = a.parse_args()
    if o.demo:
        return run(prepare(demo()), o.out + "_demo")

    import pyarrow.fs as pafs
    import pyarrow.parquet as pq
    fs = pafs.S3FileSystem()
    names = set(pq.read_schema(pb.BASE_PATH.format(year=2025).replace("s3://", ""), filesystem=fs).names)
    base_cols = [pb.CSN, pb.LABEL, pb.DATE, pb.FILING, C, M, P, pb.PAYOR_NAME, pb.PLAN_NAME]
    extra = [c for c in dict.fromkeys(CANDIDATES) if c in names]
    print("extra columns found:", extra, "| not found:", [c for c in CANDIDATES if c not in names])
    d = pe.add_dates(pb.load([2025, 2026], base_cols + extra, o.sample, fs))
    d = d[d.SVC_DATE < pd.Timestamp(o.update_start)]                       # TRAINING window only
    if o.rules_master and os.path.exists(o.rules_master):
        d = pb.add_rules_feature(d, o.rules_master, o.rules_parquet)
    run(prepare(d.reset_index(drop=True)), o.out)


if __name__ == "__main__":
    main()


#python pa_feature_entropy.py --out entropy_train 2>&1 | tee run_entropy_train.log
