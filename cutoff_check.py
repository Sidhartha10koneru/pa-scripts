"""
cutoff_check.py - choose the cut-off WITHOUT looking at the test answers, then see what it achieves on test.

For one finished run (no retraining):
  1. rebuild the run's exact validation set (same data, same windows, same seed -> same 10% of visits)
  2. score it with the run's saved model (models/lgb_full.txt)
  3. find the cut-off that catches 75 / 80 / 85 / 90% of PA services in validation
       - "validation, all"    : every validation visit (Jan 2022 - Apr 2026)
       - "validation, recent" : validation visits from --recent-start on (default Feb-Apr 2026): recent, never trained on
  4. apply each fixed cut-off to the frozen test set (scores from the run's test_preds.parquet) and report what it
     actually achieves: recall, precision, services flagged, false alarms, missed PA
     - corrected labels (passenger fix) and original labels
  "test (oracle)" rows = the cut-off read off the test answers themselves (what views 4/5 use), for comparison.

Validation labels = the run's own training labels; services = CSN + CPT (label max, score max), quality codes left out.
Run from the folder with pa_experiments.py / pa_base_adapter.py, one run per call (settings must match the run):
  python cutoff_check.py --run exp_v8d_l6pax_s42 --seed 42 --label LABEL_V6_PAX --quality-mode exclude \
     --base-path "s3://cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v8_draft/ai_prior_auth_09282026_v3_{year}_std.parquet"
  python cutoff_check.py --run exp_nofilter_2022 --seed 42 --label GROUP_LABEL --quality-mode keep \
     --base-path "s3://cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v4/ai_prior_auth_09282026_v3_{year}_std.parquet"
Output: cutoff_check/<run>.csv + printed table.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

QUALITY = r"^\d{4}F$"
TARGETS = (0.75, 0.80, 0.85, 0.90)


def norm(t):
    t["CSN"] = t.CSN.astype(str).str.replace(r"\.0$", "", regex=True)
    t["FINAL_CPT"] = t.FINAL_CPT.astype(str).str.strip().str.upper()
    return t


def services(df):
    """CSN + CPT services of a frame with columns CSN, FINAL_CPT, y, p (quality codes left out)"""
    t = norm(df[["CSN", "FINAL_CPT", "y", "p"]].rename(columns={"p": "s"}).copy())
    t = t[~t.FINAL_CPT.str.fullmatch(QUALITY)]
    return t.groupby(["CSN", "FINAL_CPT"], sort=False)[["y", "s"]].max()


def at_cut(y, s, cut):
    y, f = np.asarray(y) == 1, np.asarray(s) >= cut
    tp, fp, fn = int((f & y).sum()), int((f & ~y).sum()), int((~f & y).sum())
    return {"recall": tp / max(tp + fn, 1), "precision": tp / max(tp + fp, 1), "flagged": tp + fp,
            "TP": tp, "FP": fp, "FN": fn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--quality-mode", required=True, choices=["keep", "false", "exclude"])
    ap.add_argument("--base-path", required=True)
    ap.add_argument("--model", default="lgb_full")
    ap.add_argument("--years", nargs="+", type=int, default=[2022, 2023, 2024, 2025, 2026])
    ap.add_argument("--update-start", default="2026-02-01")
    ap.add_argument("--test-start", default="2026-05-01")
    ap.add_argument("--test-end", default="2026-08-01")
    ap.add_argument("--recent-start", default="2026-02-01")
    ap.add_argument("--no-secondary", action="store_true", help="only if the run was trained WITHOUT --include-secondary")
    ap.add_argument("--frozen", default="frozen/test_v1_2026-05_2026-07")
    ap.add_argument("--corrected-from", default="exp_v8d_l6pax_s42", help="run whose test labels are the corrected labels")
    ap.add_argument("--out", default="cutoff_check")
    a = ap.parse_args()
    # settings are read by the adapter at import time -> set them first
    os.environ.update(PA_SEED=str(a.seed), PA_LABEL_COL=a.label, PA_QUALITY_MODE=a.quality_mode, PA_BASE_PATH=a.base_path)
    sys.argv = sys.argv[:1]
    import lightgbm as lgb
    import pa_base_adapter as pb
    import pa_experiments as pe
    pb.INCLUDE_SECONDARY = not a.no_secondary
    pb.DEDUP_KEEP_TRUE = True
    os.makedirs(a.out, exist_ok=True)
    pd.set_option("display.width", 250)

    # ---- 1. the run's validation set, rebuilt exactly as pa_experiments builds it
    fs = None
    if a.base_path.startswith("s3://"):
        import pyarrow.fs as pafs
        fs = pafs.S3FileSystem()
    raw = [c for c in pb.BASE_FEATS if c not in (pb.PAYOR_F, pb.PLAN_F)] + [pb.PAYOR_NAME, pb.PLAN_NAME]
    cols = list(dict.fromkeys([pb.CSN, pb.LABEL, pb.DATE, pb.FILING] + raw))
    d = pe.add_dates(pb.load(sorted(a.years), cols, 1.0, fs))
    train, update, _ = pe.split(d, a.update_start, a.test_start, a.test_end)
    full = pd.concat([train, update], ignore_index=True)
    del d, train, update
    _, va_i = pe.val_split(full)
    va = full.iloc[va_i].copy()
    del full
    sc = json.load(open(f"{a.run}/split_counts.json")) if os.path.exists(f"{a.run}/split_counts.json") else {}
    want = sc.get(a.model, {}).get("val_lines")
    print(f"\nvalidation rebuilt: {len(va):,} lines, {va[pb.CSN].nunique():,} visits"
          + (f" | run recorded {want:,} lines -> {'MATCH' if want == len(va) else 'MISMATCH'}" if want else ""))
    if want and want != len(va):
        raise SystemExit("validation set does not match the run: check --seed/--label/--quality-mode/--base-path/--no-secondary")

    # ---- 2. score it with the saved model
    m = lgb.Booster(model_file=f"{a.run}/models/{a.model}.txt")
    cats = json.load(open(f"{a.run}/models/{a.model}_categories.json"))
    va["p"] = m.predict(pb.apply_categories(va, cats, pb.BASE_FEATS))
    first = va.groupby(pb.CSN).SVC_DATE.transform("min")
    V_all = services(va.rename(columns={pb.CSN: "CSN", pb.CPT: "FINAL_CPT"}))
    V_rec = services(va[first >= pd.Timestamp(a.recent_start)].rename(columns={pb.CSN: "CSN", pb.CPT: "FINAL_CPT"}))
    print(f"validation services: all {len(V_all):,} (PA {int(V_all.y.sum()):,}, {V_all.y.mean():.2%}) | "
          f"recent from {a.recent_start}: {len(V_rec):,} (PA {int(V_rec.y.sum()):,}, {V_rec.y.mean():.2%})")

    # ---- 3. the frozen test set with both answer keys, scored from the run's saved test predictions
    t = norm(pd.read_parquet(f"{a.run}/test_preds.parquet"))
    p = [c for c in t.columns if c.startswith(f"p_{a.model}")] or [c for c in t.columns if c.startswith("p_")]
    t = t[~t.FINAL_CPT.str.fullmatch(QUALITY)].groupby(["CSN", "FINAL_CPT"], observed=True).agg(s=(p[0], "max"))
    fz = norm(pd.read_parquet(f"{a.frozen}/test_set_services.parquet"))
    T = fz[fz.IS_QUALITY_CODE == 0].set_index(["CSN", "FINAL_CPT"])[["y"]].rename(columns={"y": "y_original"})
    c = norm(pd.read_parquet(f"{a.corrected_from}/test_preds.parquet"))
    c = c[~c.FINAL_CPT.str.fullmatch(QUALITY)].groupby(["CSN", "FINAL_CPT"], observed=True).y.max().rename("y_corrected")
    T = T.join(c, how="inner").join(t, how="inner")
    print(f"test services: {len(T):,} | PA corrected {int(T.y_corrected.sum()):,} | PA original {int(T.y_original.sum()):,}")

    # ---- 4. fixed cut-offs -> what they achieve on test
    rows = []
    for tgt in TARGETS:
        for src, D in [("validation, all", V_all), ("validation, recent", V_rec)]:
            cut = float(np.quantile(D.s[D.y == 1], 1 - tgt))
            for key in ("corrected", "original"):
                rows.append({"target": tgt, "cut-off from": src, "cut-off": cut, "labels": key,
                             **at_cut(T[f"y_{key}"], T.s, cut)})
        for key in ("corrected", "original"):
            y = T[f"y_{key}"]
            cut = float(np.quantile(T.s[y == 1], 1 - tgt))
            rows.append({"target": tgt, "cut-off from": "test (oracle)", "cut-off": cut, "labels": key, **at_cut(y, T.s, cut)})
    R = pd.DataFrame(rows).assign(run=a.run)
    R.to_csv(f"{a.out}/{a.run}.csv", index=False)
    show = R[R["labels"] == "corrected"].copy()
    show["gap"] = show.recall - show.target
    print(f"\n== {a.run}: fixed cut-off chosen before seeing the test answers -> achieved on test (corrected labels)")
    print(show[["target", "cut-off from", "cut-off", "recall", "gap", "precision", "flagged", "FP", "FN"]]
          .round({"cut-off": 3, "recall": 3, "gap": 3, "precision": 3}).to_string(index=False))
    print(f"\n(original-label rows are in {a.out}/{a.run}.csv)")


if __name__ == "__main__":
    main()
