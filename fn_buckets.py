"""
fn_buckets.py - sort every missed PA service (FN) of a trained model into 5 buckets, the same way after every run,
and keep a history so the team can see the buckets shrink from run to run.

Rewrite of fn_bucket_analysis.py (cells 1-8) as a script: same definitions, settings in one place, read from the run.

------------------------------------------------------------------------------------------------------------------
INPUT
  Needs only pandas, numpy, pyarrow (no project scripts). Reads the std claims files (S3 or local) directly.
  --run <folder>        a model run folder. Needs:
                          test_preds.parquet   (or --preds <file>) one row per TEST LINE with columns
                                               CSN, FINAL_CPT, FINAL_MODIFIERS, PLACE_OF_SERVICE_CODE,
                                               PAYOR_PLAN_CLEAN (or PAYOR + PLAN), y = label 0/1,
                                               and the model score (column p_* or --score-col)
                          split_info.json      (optional) the run's data settings; read automatically when present
  data settings         the TRAINING data the model learned from (to know each cell's history). Taken from the run's
                        split_info.json (base_path, label_col, quality_mode, years, test_start); any of them can be
                        given on the command line instead (--base-path --label --quality-mode --years --test-start
                        --include-secondary). Preparation = the training pipeline's: label TRUE/FALSE only (AMBIGUOUS
                        dropped), quality codes kept / set FALSE / excluded, secondary claims kept, repeated lines
                        (CSN+CPT+modifier+date[+filing order]) removed keeping the TRUE copy, payer/plan names cleaned.
  --frozen + --labels-from   (optional, recommended) score every run on the SAME services and the SAME answer key:
                        the frozen official test set (quality codes excluded) with the labels of a chosen run
                        (e.g. exp_v8d_l6pax_s42 = corrected labels). Without them the run's own test set and labels
                        are used, and counts move partly because the labels moved, not the model.
OUTPUT  (in <run>/fn_buckets/)
  fn_services.csv       one row per FN service: cell keys, score, training history, family, bucket
  bucket_summary.csv    per bucket: FN services, share, median score, median training rate / size, top CPTs
  run_summary.json      cut-off, recall, TP/FP/FN/TN, settings used
  + one row appended to  fn_bucket_history.csv  (in the working folder): run, date, FN per bucket -> compare runs
  cell history is cached in  fn_bucket_cache/  per data setting, so runs on the same data reload nothing heavy.

RUN
  python fn_buckets.py --run exp_v8d_l6pax_s42 --frozen frozen/test_v1_2026-05_2026-07 --labels-from exp_v8d_l6pax_s42
  python fn_buckets.py --run exp_nofilter_2022 --label GROUP_LABEL --quality-mode keep \
        --base-path "s3://.../std_data_v4/ai_prior_auth_09282026_v3_{year}_std.parquet" \
        --frozen frozen/test_v1_2026-05_2026-07 --labels-from exp_v8d_l6pax_s42
  python fn_buckets.py --history                      # print the history table only

------------------------------------------------------------------------------------------------------------------
DEFINITIONS (data-driven unless marked [ASSUMPTION])
  service      CSN + CPT. PA if any of its lines is TRUE; score = highest line score; its cell = the modifier,
               place of service and payer|plan of its highest-scoring line.
  cell         CPT + modifier + place of service + payer|plan (the model's 4 categorical inputs per CPT).
  cut-off      score that catches --recall (default 80%) of PA services, read off the test set itself
               (same as the benchmark analysis; a model-comparison setting, not a production cut-off).
  FN           PA service scored below the cut-off.
  training history of a cell: lines before the test window, after the run's own label preparation
               (AMBIGUOUS dropped, repeats removed keeping the TRUE copy, quality codes per --quality-mode).
BUCKETS (first rule that matches wins; thresholds are flags)
  1 likely label noise   quality code, or an E/M / lab service on a visit that also has a PA "anchor" service
                         [ASSUMPTION: heuristic by CPT family, does NOT check auth numbers]
  2 unseen or rare cell  fewer than --min-n (20) training lines for the exact cell        [threshold chosen by us]
  3 historically non-PA  training PA rate < --low (5%)                                    [threshold chosen by us]
  4 mixed cell           training PA rate between --low and --high (5% - 50%)             [threshold chosen by us]
  5 model miss           training PA rate >= --high (50%) yet scored below the cut-off
CPT FAMILIES  [ASSUMPTION: from the CPT/HCPCS code structure, general coding knowledge, not from our data;
               to be reviewed by the coding team]  -> FAMILIES below
------------------------------------------------------------------------------------------------------------------
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import re

import numpy as np
import pandas as pd

KEY = ["FINAL_CPT", "FINAL_MODIFIERS", "PLACE_OF_SERVICE_CODE", "PAYOR_PLAN_CLEAN"]
QUALITY = r"\d{4}F"
# [ASSUMPTION] CPT families by code range (first match wins). Edit here, never inside the logic.
FAMILIES = [
    ("quality", r"\d{4}F"),                      # CPT Category II performance-measure codes (no payment, never PA)
    ("E/M",     r"99[2-4]\d\d|G2211"),           # evaluation & management 99200-99499 + G2211 visit-complexity add-on
    ("lab",     r"8\d{4}|36415"),                # pathology & laboratory 80000-89999 + 36415 venipuncture
    ("anchor",  r"7\d{4}|[1-6]\d{4}|J\d{4}|97\d{3}"),  # radiology 7xxxx, surgery 10000-69999, J-code drugs, 97xxx PT/rehab
]
NOISE_FAMILIES = ("E/M", "lab")                  # [ASSUMPTION] families that ride along with an anchor's auth
BUCKETS = ["1 likely label noise", "2 unseen or rare cell", "3 historically non-PA", "4 mixed cell", "5 model miss"]
HISTORY = "fn_bucket_history.csv"


# ---------------------------------------------------------------------------------------------------- settings
def settings(a):
    """data settings of the run: split_info.json, overridden by command-line flags"""
    info = {}
    p = f"{a.run}/split_info.json"
    if os.path.exists(p):
        info = json.load(open(p))
    s = {"base_path": a.base_path or info.get("base_path"),
         "label_col": a.label or info.get("label_col", "GROUP_LABEL"),
         "quality_mode": a.quality_mode or info.get("quality_mode", "keep"),
         "years": a.years or info.get("years"),
         "test_start": a.test_start or info.get("test_start"),
         "include_secondary": info.get("include_secondary", True) if a.include_secondary is None else a.include_secondary}
    missing = [k for k, v in s.items() if v in (None, [], "")]
    if missing:
        raise SystemExit(f"missing data settings {missing}: not in {p}, give them on the command line")
    return s


# ---------------------------------------------------------------------------------------------------- data preparation
# Self-contained copy of the training-data preparation (same rules as the training pipeline), so this script
# needs only pandas / numpy / pyarrow. Column names of the std claims files:
CSN, LABEL_DEFAULT, DATE, FILING, CPT, MOD, POS, PAYOR, PLAN = ("CSN", "GROUP_LABEL", "SERVICEDATE", "FILING_ORDER_C",
    "FINAL_CPT", "FINAL_MODIFIERS", "PLACE_OF_SERVICE_CODE", "PAYOR", "PLAN")


def clean_name(x):
    """upper-case, squeeze spaces, drop date suffixes like 'AS OF 01/01/2026' (same as training)"""
    x = x.astype(str).str.upper().str.replace(r"\bAS OF\b.*$", "", regex=True)
    return x.str.replace(r"\s+", " ", regex=True).str.strip()


def to_label(x):
    """TRUE/1 -> 1, FALSE/0 -> 0, anything else (AMBIGUOUS, blank) -> NaN = dropped"""
    return x.astype(str).str.strip().str.upper().map({"TRUE": 1, "1": 1, "1.0": 1, "FALSE": 0, "0": 0, "0.0": 0})


def prep_year(d, label, quality_mode, include_secondary):
    """one year of raw lines -> training lines, exactly as the model saw them"""
    d = d.loc[:, ~d.columns.duplicated()].copy()
    d["y"] = to_label(d[label])
    if quality_mode != "keep":                                    # quality codes = CPT Category II (####F)
        q = d[CPT].astype(str).str.strip().str.upper().str.fullmatch(QUALITY)
        d = d[~q] if quality_mode == "exclude" else d.assign(y=d.y.where(~q, 0))
    d = d[d.y.notna()]                                            # AMBIGUOUS / missing label dropped
    keys = [CSN, CPT, MOD, DATE]
    if FILING in d:
        if include_secondary:
            keys.append(FILING)                                   # secondary claims kept as their own rows
        else:
            d = d[pd.to_numeric(d[FILING], errors="coerce") == 1]  # primary claims only
    d = d.sort_values("y", ascending=False, kind="stable").drop_duplicates(keys).sort_index()  # repeats: TRUE copy wins
    d["y"] = d.y.astype(int)
    d["PAYOR_CLEAN"] = clean_name(d[PAYOR])
    d["PAYOR_PLAN_CLEAN"] = d.PAYOR_CLEAN + " | " + clean_name(d[PLAN])
    d["SVC_DATE"] = pd.to_datetime(d[DATE], errors="coerce")
    return d[d.SVC_DATE.notna()]


# ---------------------------------------------------------------------------------------------------- training history
def cell_history(s, cache_dir="fn_bucket_cache"):
    """training PA history per cell and per CPT, for one data setting (cached: same data -> no reload)"""
    tag = hashlib.sha256(json.dumps(s, sort_keys=True).encode()).hexdigest()[:12]
    f_cell, f_cpt = f"{cache_dir}/cells_{tag}.parquet", f"{cache_dir}/cpts_{tag}.parquet"
    if os.path.exists(f_cell) and os.path.exists(f_cpt):
        print(f"[train] cell history from cache {f_cell}")
        return pd.read_parquet(f_cell), pd.read_parquet(f_cpt)
    fs, ts = None, pd.Timestamp(s["test_start"])
    if s["base_path"].startswith("s3://"):
        import pyarrow.fs as pafs
        fs = pafs.S3FileSystem()
    cols = [CSN, s["label_col"], DATE, FILING, CPT, MOD, POS, PAYOR, PLAN]
    cells, cpts = [], []
    for y in sorted(s["years"]):
        path = s["base_path"].format(year=y)
        if fs is not None:
            path = path.replace("s3://", "")
        d = pd.read_parquet(path, columns=list(dict.fromkeys(cols)), filesystem=fs)
        tr = prep_year(d, s["label_col"], s["quality_mode"], s["include_secondary"])
        tr = tr[tr.SVC_DATE < ts]
        for c in KEY:
            tr[c] = tr[c].astype(object).fillna("").astype(str).str.strip()
        cells.append(tr.groupby(KEY).y.agg(["sum", "size"]))
        cpts.append(tr.groupby("FINAL_CPT").y.agg(["sum", "size"]))
        print(f"[train] {y}: {len(d):,} raw lines -> {len(tr):,} training lines before {s['test_start']}")
        del d, tr
    c = pd.concat(cells).groupby(level=list(range(len(KEY)))).sum()
    cell = pd.DataFrame({"train_n": c["size"], "train_pa_rate": c["sum"] / c["size"]}).reset_index()
    p = pd.concat(cpts).groupby(level=0).sum()
    cpt = pd.DataFrame({"train_cpt_n": p["size"], "train_cpt_pa_rate": p["sum"] / p["size"]}).reset_index()
    print(f"[train] {int(c['size'].sum()):,} training lines | PA rate {c['sum'].sum() / c['size'].sum():.2%} | {len(cell):,} cells")
    os.makedirs(cache_dir, exist_ok=True)
    cell.to_parquet(f_cell, index=False); cpt.to_parquet(f_cpt, index=False)
    return cell, cpt


# ---------------------------------------------------------------------------------------------------- test services
def test_services(run, frozen=None, labels_from=None, score_col=None, preds="test_preds.parquet"):
    """CSN + CPT services: label = max, score = max, cell keys of the highest-scoring line"""
    t = pd.read_parquet(f"{run}/{preds}")
    t = t.loc[:, ~t.columns.duplicated()]
    p = [score_col] if score_col else [c for c in t.columns if c.startswith("p_")]
    if len(p) != 1 or p[0] not in t:
        raise SystemExit(f"{run}/{preds}: need one score column (p_* or --score-col), found {p}")
    if "PAYOR_PLAN_CLEAN" not in t and {PAYOR, PLAN} <= set(t.columns):
        t["PAYOR_PLAN_CLEAN"] = clean_name(t[PAYOR]) + " | " + clean_name(t[PLAN])
    need = ["CSN", "y"] + KEY
    if [c for c in need if c not in t]:
        raise SystemExit(f"{run}/{preds}: missing columns {[c for c in need if c not in t]} (see INPUT in the docstring)")
    t["CSN"] = t.CSN.astype(str).str.replace(r"\.0$", "", regex=True)
    for c in KEY:
        t[c] = t[c].astype(object).fillna("").astype(str).str.strip()
    s = t.groupby(["CSN", "FINAL_CPT"]).agg(y=("y", "max"), score=(p[0], "max")).reset_index()
    top = t.loc[t.groupby(["CSN", "FINAL_CPT"])[p[0]].idxmax(), ["CSN"] + KEY]
    s = s.merge(top, on=["CSN", "FINAL_CPT"])
    src = "run's own test set and labels"
    if frozen:                                     # same services for every run (quality codes excluded)
        fz = pd.read_parquet(f"{frozen}/test_set_services.parquet")
        fz["CSN"] = fz.CSN.astype(str).str.replace(r"\.0$", "", regex=True)
        fz["FINAL_CPT"] = fz.FINAL_CPT.astype(str).str.strip()
        keep = fz[fz.IS_QUALITY_CODE == 0][["CSN", "FINAL_CPT", "y"]]
        if labels_from:                            # same answer key for every run
            lab = pd.read_parquet(f"{labels_from}/test_preds.parquet", columns=["CSN", "FINAL_CPT", "y"])
            lab["CSN"] = lab.CSN.astype(str).str.replace(r"\.0$", "", regex=True)
            lab["FINAL_CPT"] = lab.FINAL_CPT.astype(str).str.strip()
            keep = keep[["CSN", "FINAL_CPT"]].merge(lab.groupby(["CSN", "FINAL_CPT"]).y.max().reset_index(),
                                                    on=["CSN", "FINAL_CPT"])
            src = f"frozen set {frozen}, labels of {labels_from}"
        else:
            src = f"frozen set {frozen}, its original labels"
        missing = len(keep) - len(keep.merge(s[["CSN", "FINAL_CPT"]], on=["CSN", "FINAL_CPT"]))
        s = keep.merge(s.drop(columns="y"), on=["CSN", "FINAL_CPT"])
        if missing:
            print(f"[test] {missing:,} frozen services not scored by this run (left out)")
    print(f"[test] {len(s):,} services | PA {int(s.y.sum()):,} | {src}")
    return s, src


# ---------------------------------------------------------------------------------------------------- FN + buckets
def family(cpt):
    c = str(cpt).upper()
    for name, pat in FAMILIES:
        if re.fullmatch(pat, c):
            return name
    return "other"


def find_fn(s, recall, cutoff=None):
    cut = float(cutoff) if cutoff is not None else float(np.quantile(s.score[s.y == 1], 1 - recall))
    f, y = s.score >= cut, s.y == 1
    conf = {"cutoff": cut, "TP": int((f & y).sum()), "FP": int((f & ~y).sum()), "FN": int((~f & y).sum()),
            "TN": int((~f & ~y).sum())}
    conf["recall"] = conf["TP"] / max(conf["TP"] + conf["FN"], 1)
    conf["precision"] = conf["TP"] / max(conf["TP"] + conf["FP"], 1)
    return s[y & ~f].copy(), conf


def assign_buckets(fn, s, cell, cpt, min_n, low, high):
    fn = fn.merge(cell, on=KEY, how="left").merge(cpt, on="FINAL_CPT", how="left")
    fn["train_n"] = fn.train_n.fillna(0)
    fn["family"] = fn.FINAL_CPT.map(family)
    pa = s[s.y == 1].assign(family=lambda x: x.FINAL_CPT.map(family))
    anchor_visits = set(pa.loc[pa.family == "anchor", "CSN"])
    fn["anchor_in_visit"] = fn.CSN.isin(anchor_visits) & fn.family.ne("anchor")
    fn["bucket"] = np.select(
        [fn.family.eq("quality") | (fn.family.isin(NOISE_FAMILIES) & fn.anchor_in_visit),
         fn.train_n < min_n, fn.train_pa_rate < low, fn.train_pa_rate < high],
        BUCKETS[:4], BUCKETS[4])
    return fn


def summarise(fn):
    b = fn.groupby("bucket").agg(fn_services=("CSN", "size"), median_score=("score", "median"),
                                 median_train_rate=("train_pa_rate", "median"), median_train_n=("train_n", "median"))
    b = b.reindex(BUCKETS).fillna({"fn_services": 0})
    b["share"] = b.fn_services / max(b.fn_services.sum(), 1)
    b["top_cpts"] = [", ".join(f"{k} ({v})" for k, v in fn[fn.bucket == k].FINAL_CPT.value_counts().head(6).items())
                     for k in b.index]
    return b.reset_index()


# ---------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--run")
    ap.add_argument("--frozen", default=None, help="frozen test-set folder (same services for every run)")
    ap.add_argument("--labels-from", default=None, help="run whose test labels are the answer key (with --frozen)")
    ap.add_argument("--preds", default="test_preds.parquet", help="file in the run folder with test lines + scores")
    ap.add_argument("--score-col", default=None, help="score column (default: the one column starting with p_)")
    ap.add_argument("--recall", type=float, default=0.80)
    ap.add_argument("--cutoff", type=float, default=None,
                    help="fixed score cut-off instead of --recall (e.g. 0.8): FN then varies between runs")
    ap.add_argument("--min-n", type=int, default=20)
    ap.add_argument("--low", type=float, default=0.05)
    ap.add_argument("--high", type=float, default=0.50)
    ap.add_argument("--base-path"); ap.add_argument("--label"); ap.add_argument("--quality-mode")
    ap.add_argument("--years", nargs="+", type=int); ap.add_argument("--test-start")
    ap.add_argument("--include-secondary", type=lambda v: v.lower() in ("1", "true", "yes"), default=None)
    ap.add_argument("--history", action="store_true", help="only print the history of all runs")
    a = ap.parse_args()
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70)
    if a.history or not a.run:
        print(pd.read_csv(HISTORY).to_string(index=False) if os.path.exists(HISTORY) else f"no {HISTORY} yet")
        return

    s_set = settings(a)
    cell, cpt = cell_history(s_set)
    s, src = test_services(a.run, a.frozen, a.labels_from, a.score_col, a.preds)
    fn, conf = find_fn(s, a.recall, a.cutoff)
    fn = assign_buckets(fn, s, cell, cpt, a.min_n, a.low, a.high)
    summ = summarise(fn)

    out = f"{a.run}/fn_buckets"
    os.makedirs(out, exist_ok=True)
    fn.to_csv(f"{out}/fn_services.csv", index=False)
    summ.to_csv(f"{out}/bucket_summary.csv", index=False)
    meta = {"run": a.run, "created": dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "test": src, **conf,
            "recall_target": a.recall, "min_n": a.min_n, "low": a.low, "high": a.high, "data": s_set}
    json.dump(meta, open(f"{out}/run_summary.json", "w"), indent=1, default=str)

    print(f"\ncut-off {conf['cutoff']:.4f} ({'fixed' if a.cutoff is not None else f'catches {a.recall:.0%} of PA'}) | TP {conf['TP']:,} FP {conf['FP']:,} "
          f"FN {conf['FN']:,} TN {conf['TN']:,} | precision {conf['precision']:.1%}")
    print("\n== FN by bucket\n" + summ.round(3).to_string(index=False))

    row = {"run": a.run, "created": meta["created"], "test": src,
           "cutoff_rule": f"fixed {a.cutoff}" if a.cutoff is not None else f"recall {a.recall:.0%}", "labels_trained": s_set["label_col"],
           "quality_mode": s_set["quality_mode"], "recall_target": a.recall if a.cutoff is None else None, "fixed_cutoff": a.cutoff,
           "cutoff": round(conf["cutoff"], 4), "recall": round(conf["recall"], 4),
           "PA_services": conf["TP"] + conf["FN"], "FN": conf["FN"], "FP": conf["FP"],
           **{b: int(summ.set_index("bucket").fn_services.get(b, 0)) for b in BUCKETS}}
    h = pd.read_csv(HISTORY) if os.path.exists(HISTORY) else pd.DataFrame()
    if len(h):
        same = (h.run == a.run) & (h.test == src) & (h.cutoff_rule == row["cutoff_rule"]) if "cutoff_rule" in h else False
        h = h[~same]                                                                   # rerun replaces its row
    h = pd.concat([h, pd.DataFrame([row])], ignore_index=True)
    h.to_csv(HISTORY, index=False)
    print(f"\n== history ({HISTORY}, rows on the same test set + labels are comparable)")
    print(h[(h.test == src) & (h.cutoff_rule == row["cutoff_rule"])].drop(columns=["test"]).to_string(index=False))
    print(f"\noutputs -> {out}/ (fn_services.csv, bucket_summary.csv, run_summary.json)")


if __name__ == "__main__":
    main()
