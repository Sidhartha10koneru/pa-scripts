"""
register_run.py - register a finished pa_experiments run in the model registry.

For one run folder it:
  1. builds the run ID from the run's settings (naming convention below), or takes --run-id
  2. scores the run on the FROZEN official test set (same services for every run) and checks coverage
  3. writes <run>/run_card.json  (settings, code fingerprint, split counts, official metrics, notes, status)
  4. adds / replaces the run's row in registry.csv
  5. --push: copies the run folder to  <s3>/runs/<run_id>/  and registry.csv to  <s3>/registry.csv

Naming convention  pa-<model>-<data>-<labels>-<qc>-<val>-s<seed>-<train years>-<test window>-<yyyymmdd>
  model   lgb5          LightGBM, 5 base features (+feature tag if --extra-feats, e.g. lgb5+dept)
  data    v4 / v6       dataset version read
  labels  grp / l6      GROUP_LABEL / LABEL_V6
  qc      qckeep / qcfalse / qcexcl     quality-code handling
  val     rv10 / tv<yyyymm>             random 10% visits / time-based from that month (+ref if refit)
  seed    s42           training + validation seed
  years   y22-26        first-last training year
  test    t2605-2607    test window (first-last month)
  e.g.    pa-lgb5-v4-grp-qcexcl-rv10-s42-y22-26-t2605-2607-20261104

Status: benchmark | candidate | archived (one benchmark at a time; registering a new benchmark archives the old one).

Run:
  python register_run.py exp_qc_exclude_rv --frozen frozen/test_v1_2026-05_2026-07 --status candidate \
         --notes "quality codes excluded; bootstrap + seed 7 confirm AUC-PR +0.008"
  python register_run.py exp_nofilter_2022 --frozen ... --status benchmark --qc qckeep --seed 42   # old runs: override
"""
import argparse
import hashlib
import json
import os
import re
import subprocess

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

QUALITY = r"^\d{4}F$"
S3 = "s3://cli-ml-data-prod/datalake/ai-prior-auth/09282026/model_registry"


def jload(p):
    return json.load(open(p)) if os.path.exists(p) else {}


def run_id(info, a, created):
    data = "v6" if "v6" in str(info.get("base_path", "")) else "v4"
    labels = "l6" if info.get("label_col", "GROUP_LABEL") == "LABEL_V6" else "grp"
    qc = a.qc or {"keep": "qckeep", "false": "qcfalse", "exclude": "qcexcl"}.get(info.get("quality_mode", "keep"))
    if data == "v6" and labels == "l6" and qc == "qckeep":
        qc = "qcfalse"                                             # v6 LABEL_V6 = quality codes FALSE
    vs = info.get("val_start")
    val = f"tv{vs[:7].replace('-', '')}" if vs else "rv10"
    if info.get("refit_full"):
        val += "ref"
    seed = a.seed if a.seed is not None else info.get("seed", 42)
    yrs = info.get("years", [])
    ty = f"y{str(min(yrs))[2:]}-{str(max(yrs))[2:]}" if yrs else "y?"
    ts, te = info.get("test_start", "????-??"), info.get("test_end", "????-??")
    te_m = (pd.Timestamp(te) - pd.Timedelta(days=1)).strftime("%y%m") if te[0].isdigit() else "????"
    test = f"t{ts[2:4]}{ts[5:7]}-{te_m}"
    model = a.model_tag or "lgb5"
    return f"pa-{model}-{data}-{labels}-{qc}-{val}-s{seed}-{ty}-{test}-{created}"


def services(path):
    t = pd.read_parquet(path)
    t = t.loc[:, ~t.columns.duplicated()]
    p = [c for c in t.columns if c.startswith("p_")]
    if not p:
        raise SystemExit(f"{path}: no p_* score column")
    t["CSN"] = t.CSN.astype(str)
    t["FINAL_CPT"] = t.FINAL_CPT.astype(str).str.strip().str.upper()
    return t.groupby(["CSN", "FINAL_CPT"], observed=True).agg(score=(p[0], "max")).reset_index(), p[0]


def metrics(y, s):
    y, s = np.asarray(y), np.asarray(s)
    r = {"AUC_PR": average_precision_score(y, s), "AUC_ROC": roc_auc_score(y, s)}
    for rec in (0.80, 0.90):
        cut = np.quantile(s[y == 1], 1 - rec)
        f = s >= cut
        k = int(rec * 100)
        tp, fp = int((f & (y == 1)).sum()), int((f & (y == 0)).sum())
        r |= {f"FP@{k}": fp, f"TP@{k}": tp, f"FN@{k}": int((~f & (y == 1)).sum()), f"precision@{k}": tp / max(tp + fp, 1)}
    best = max(((t, 2 * ((s >= t) & (y == 1)).sum() / max((s >= t).sum() + y.sum(), 1))
                for t in np.arange(0.05, 1.0, 0.05)), key=lambda z: z[1])
    f = s >= best[0]
    r |= {"best_thr": round(float(best[0]), 2), "TP@best": int((f & (y == 1)).sum()), "FP@best": int((f & (y == 0)).sum()),
          "FN@best": int((~f & (y == 1)).sum()), "F1@best": float(best[1])}
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="run folder (pa_experiments --out)")
    ap.add_argument("--frozen", required=True, help="frozen test-set folder (freeze_test_set.py --out)")
    ap.add_argument("--registry", default="registry.csv")
    ap.add_argument("--status", choices=["benchmark", "candidate", "archived"], default="candidate")
    ap.add_argument("--notes", default="")
    ap.add_argument("--run-id", default=None, help="override the generated ID")
    ap.add_argument("--qc", choices=["qckeep", "qcfalse", "qcexcl"], default=None, help="override (runs made before provenance)")
    ap.add_argument("--seed", type=int, default=None, help="override (runs made before provenance)")
    ap.add_argument("--model-tag", default=None, help="e.g. lgb5+dept when extra features were used")
    ap.add_argument("--push", action="store_true", help="copy run folder + registry to S3")
    ap.add_argument("--s3", default=S3)
    a = ap.parse_args()

    info = jload(f"{a.run}/split_info.json")
    counts = jload(f"{a.run}/split_counts.json")
    created = pd.Timestamp(os.path.getmtime(f"{a.run}/test_preds.parquet"), unit="s").strftime("%Y%m%d")
    rid = a.run_id or run_id(info, a, created)

    # score on the frozen official test set
    fz = pd.read_parquet(f"{a.frozen}/test_set_services.parquet")
    fz["CSN"] = fz.CSN.astype(str)
    man = jload(f"{a.frozen}/manifest.json")
    off = fz[fz.IS_QUALITY_CODE == 0][["CSN", "FINAL_CPT", "y"]]
    sv, score_col = services(f"{a.run}/test_preds.parquet")
    m = off.merge(sv, on=["CSN", "FINAL_CPT"], how="left")
    missing = int(m.score.isna().sum())
    extra = len(sv[~sv.FINAL_CPT.str.fullmatch(QUALITY)]) - (len(m) - missing)
    coverage = 1 - missing / len(m)
    print(f"{rid}\n  frozen official services {len(m):,} (PA {int(m.y.sum()):,}) | scored {len(m) - missing:,} "
          f"| missing {missing:,} | run's extra non-quality services {extra:,}")
    mm = m.dropna(subset=["score"])
    met = metrics(mm.y, mm.score)
    vis = mm.groupby("CSN").agg(y=("y", "max"), score=("score", "max"))
    met |= {f"visit_{k}": v for k, v in metrics(vis.y, vis.score).items() if k in ("AUC_PR", "FP@80", "precision@80")}

    code = info.get("code_sha256") or {f: hashlib.sha256(open(f"{a.run}/code/{f}", "rb").read()).hexdigest()[:12]
                                       for f in (os.listdir(f"{a.run}/code") if os.path.isdir(f"{a.run}/code") else [])}
    card = {"run_id": rid, "folder": os.path.abspath(a.run), "status": a.status, "notes": a.notes,
            "registered": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"), "score_column": score_col,
            "settings": {k: info.get(k) for k in ["command", "env", "base_path", "label_col", "quality_mode", "seed", "years",
                                                  "update_start", "test_start", "test_end", "include_secondary",
                                                  "secondary_in_test", "dedup_keep_true", "val_start", "refit_full",
                                                  "es_metric", "models"]},
            "code_sha256": code or "not recorded (run made before provenance was added)",
            "split_counts": counts,
            "test_set": {"name": man.get("name"), "sha256": man.get("sha256"), "official_services": len(m),
                         "official_PA": int(m.y.sum()), "scored": len(m) - missing, "coverage": round(coverage, 5)},
            "official_metrics": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in met.items()}}
    json.dump(card, open(f"{a.run}/run_card.json", "w"), indent=1, default=str)

    tr = (counts.get(next(iter(counts), ""), {}) if counts else {})
    row = {"run_id": rid, "status": a.status, "registered": card["registered"], "folder": a.run,
           "data": rid.split("-")[2], "labels": rid.split("-")[3], "qc": rid.split("-")[4], "val": rid.split("-")[5],
           "seed": rid.split("-")[6], "fit_lines": tr.get("train_lines"), "val_lines": tr.get("val_lines"),
           "test_set": man.get("name"), "coverage": round(coverage, 5),
           **{k: card["official_metrics"][k] for k in ["AUC_PR", "AUC_ROC", "FP@80", "precision@80", "FP@90", "precision@90",
                                                       "best_thr", "TP@best", "FP@best", "FN@best", "visit_AUC_PR"]},
           "code": ";".join(f"{os.path.basename(k)}:{v}" for k, v in code.items()) if isinstance(code, dict) else "",
           "notes": a.notes}
    R = pd.read_csv(a.registry) if os.path.exists(a.registry) else pd.DataFrame()
    if len(R):
        R = R[R.run_id != rid]
        if a.status == "benchmark":
            R.loc[R.status == "benchmark", "status"] = "archived"
    R = pd.concat([R, pd.DataFrame([row])], ignore_index=True)
    R.to_csv(a.registry, index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(pd.DataFrame([row]).drop(columns=["notes", "code"]).T.to_string(header=False))
    if coverage < 0.999:
        print(f"!! coverage {coverage:.3%}: the run did not score all frozen test services - check its test window/filters")
    print(f"\nrun_card -> {a.run}/run_card.json | registry -> {a.registry} ({len(R)} runs)")

    dest = f"{a.s3.rstrip('/')}/runs/{rid}/"
    if a.push:
        subprocess.run(["aws", "s3", "cp", "--recursive", a.run, dest], check=True)
        subprocess.run(["aws", "s3", "cp", a.registry, f"{a.s3.rstrip('/')}/registry.csv"], check=True)
        print(f"pushed -> {dest}")
    else:
        print(f"to push: aws s3 cp --recursive {a.run} {dest} && aws s3 cp {a.registry} {a.s3.rstrip('/')}/registry.csv")


if __name__ == "__main__":
    main()
