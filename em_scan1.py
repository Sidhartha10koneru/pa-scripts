"""
em_scan.py - what separates E/M PA (TP) from E/M false alarms (FP)?  Scans every usable column of the data file.

Groups (at the cut-off for --recall on ALL test services, same as the headline):
  TP = E/M services flagged and PA        FP = E/M services flagged and not PA
  --scope top5 (default): only the 5 plans with the most E/M FP  -> differences WITHIN those plans
  --scope all:            all E/M services

For every column it scores how well the column alone separates TP from FP:
  AUC  0.5 = no information, 1.0 = perfect. Categorical columns are target-encoded with 2-fold cross-fitting
       (rates learned on one half, scored on the other) so the AUC is not flattered; numeric columns are used as is
       (AUC shown as max(auc, 1-auc), with the direction).
  fill_TP / fill_FP  share of services where the column is filled (a big gap = filled only in some situations)
Reference rows: the model's own score, and the model's features (they should be weak here - TP and FP look alike).

EXCLUDED automatically (listed in the log for review):
  - identifiers / personal fields (patient, member, names, address, AccuReg PID/PD1/NK1/GT1/IN1/IN2/IN3 ...)
  - protected attributes (race, ethnicity, sex, gender ...) - never candidate features
  - anything only known AFTER the service (auth number, labels, denial / CARC / RARC / Acuity codes, payment ...)
  - dates
Raw values are printed only for low-cardinality columns (<= 40 values); everything stays on EFS.

  nohup python -u em_scan.py --run exp_feat_core_s42 --out em_scan_core > run_em_scan.log 2>&1 &
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.fs as pafs
from sklearn.metrics import roc_auc_score

DATA = "s3://cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v8_draft/ai_prior_auth_09282026_v3_2026_std.parquet"
EXCLUDE = {
    "identifier / personal": r"PAT_?ID|PATIENT|MRN|SSN|(?<!PAYOR_)(?<!PLAN_)NAME|PAT_.*NAME|PCPNAME|PROVIDER_NAME|DOB|BIRTH|ADDR|PHONE|EMAIL|ZIP|POSTAL|MEMBER|SUBSCRIB|GUARANT|"
                             r"^PID|^PD1|^NK1|^GT1|^IN1|^IN2|^IN3|_PID_|_NK1_|_GT1_|_IN1_|_IN2_|_IN3_|^CSN$|SRC_ROW|SRC_YEAR",
    "protected attribute": r"RACE|ETHNIC|SEX|GENDER|RELIGION|MARITAL|LANGUAGE",
    "known after service": r"AUTH|LABEL|CARC|RARC|DENIAL|DENY|PAID|PAYMENT|ADJUST|ADJ_|REMIT|835|STATUS|ACUITY|RELABEL|PASSENGER|"
                           r"VISIT_LEVEL|REASON|ALLOWED|BALANCE|WRITE_?OFF|RECOUP|APPEAL|EOB|CHECK_|GROUP_LABEL|_CODES?$|"
                           r"^N_TRUE$|^N_FALSE$|^N_AMBIGUOUS$|^GROUP_SIZE$|^GROUP_GUID$|INVOICE|SUBMISSION",
}
PRINT_VALUES_MAX = 40


def key(s):
    return s.astype(object).where(s.notna(), None).map(lambda v: "" if v is None else str(v)) \
        .str.strip().str.replace(r"^(\d+)\.0$", r"\1", regex=True)


def is_em(cpt):
    n = pd.to_numeric(cpt.where(cpt.str.fullmatch(r"\d{5}")), errors="coerce")
    return n.between(99202, 99499)


ALLOW = {"DX_NAME", "ENCOUNTER_DEPARTMENT_NAME", "BUCKET_PAYOR_NAME"}      # descriptions, not people


def excluded(col):
    if col.upper() in ALLOW:
        return None
    for why, pat in EXCLUDE.items():
        if re.search(pat, col.upper()):
            return why
    return None


def cat_auc(v, y, m=20, seed=0):
    """2-fold cross-fitted target encoding -> AUC"""
    v = key(v).replace("", "(missing)").values
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, 2, len(v))
    enc = np.zeros(len(v))
    for f in (0, 1):
        tr = fold != f
        prior = y[tr].mean()
        g = pd.DataFrame({"v": v[tr], "y": y[tr]}).groupby("v").y.agg(["sum", "count"])
        rate = (g["sum"] + m * prior) / (g["count"] + m)
        enc[~tr] = pd.Series(v[~tr]).map(rate).fillna(prior).values
    return roc_auc_score(y, enc)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--recall", type=float, default=0.85)
    ap.add_argument("--scope", choices=["top5", "all"], default="top5")
    ap.add_argument("--plan-col", default="PAYOR_PLAN_CLEAN")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20); pd.set_option("display.max_colwidth", 50)

    t = pd.read_parquet(f"{a.run}/test_preds.parquet")
    t = t.loc[:, ~t.columns.duplicated()]
    pcol = [c for c in t.columns if c.startswith("p_")][0]
    t["k_csn"], t["k_cpt"] = key(t.CSN), key(t.FINAL_CPT).str.upper()
    S = t.groupby(["k_csn", "k_cpt"]).agg(y=("y", "max"), s=(pcol, "max")).reset_index()
    S["y"] = S.y.astype(bool)
    cut = float(np.quantile(S.s[S.y], 1 - a.recall))
    # E/M flagged services; keep the top-scoring line of each (its SRC_ROW is used to fetch that line's columns)
    t = t[is_em(t.k_cpt)].merge(S[S.s >= cut][["k_csn", "k_cpt", "y", "s"]].rename(columns={"y": "y_svc", "s": "s_svc"}),
                                on=["k_csn", "k_cpt"])
    L = t.sort_values(pcol, ascending=False).drop_duplicates(["k_csn", "k_cpt"]).reset_index(drop=True)
    # visit-level context from all test services (no extra data needed)
    vis = S.assign(em=is_em(S.k_cpt)).groupby("k_csn").agg(V_N_SERVICES=("k_cpt", "size"), V_N_NON_EM=("em", lambda s: int((~s).sum())))
    L = L.join(vis, on="k_csn")
    if a.scope == "top5":
        fp_plans = L[~L.y_svc].groupby(a.plan_col).size().sort_values(ascending=False).head(5).index
        L = L[L[a.plan_col].isin(fp_plans)].reset_index(drop=True)
        print("scope: top-5 FP plans:", list(fp_plans))
    y = L.y_svc.values.astype(int)
    print(f"cut-off {cut:.6f} | E/M flagged services: TP {y.sum():,} | FP {(1 - y).sum():,}")
    if y.sum() < 20 or (1 - y).sum() < 20:
        raise SystemExit("too few TP or FP in this scope to compare - try --scope all")

    # data file: columns, exclusions, fetch the exact lines by SRC_ROW in batches
    fs, p = (pafs.S3FileSystem(), a.data.replace("s3://", "")) if a.data.startswith("s3://") else (pafs.LocalFileSystem(), os.path.abspath(a.data))
    dset = ds.dataset(p, filesystem=fs, format="parquet")
    schema = dset.schema
    ex = {c: excluded(c) for c in schema.names}
    ex.update({c: "date" for c in schema.names if not ex[c] and str(schema.field(c).type).startswith(("timestamp", "date"))})
    keep = [c for c in schema.names if not ex[c]]
    print(f"\ncolumns in file {len(schema.names)} | scanned {len(keep)} | excluded {sum(bool(v) for v in ex.values())}")
    for why in sorted({v for v in ex.values() if v}):
        print(f"  excluded ({why}): {sorted(c for c, v in ex.items() if v == why)}")
    rows = pd.to_numeric(L.SRC_ROW, errors="coerce").astype("Int64")
    csn_file = key(dset.to_table(columns=["CSN"]).column("CSN").to_pandas())
    off = next((o for o in (0, -1, 1) if (csn_file.reindex((rows + o).astype(float)).values == L.k_csn.values).mean() > 0.99), None)
    if off is None:
        raise SystemExit("SRC_ROW does not line up with the data file rows - is --data the same 2026 file the run used?")
    idx = (rows + off).astype(int).values
    print(f"lines fetched by SRC_ROW (offset {off})")

    res, values = [], {}
    feats_model = [c for c in ["FINAL_CPT", "FINAL_MODIFIERS", "PLACE_OF_SERVICE_CODE", "PAYOR_CLEAN", "PAYOR_PLAN_CLEAN",
                               "Primary_DX", "DX3"] if c in L]
    res.append({"column": "MODEL SCORE (reference)", "kind": "numeric", "AUC": roc_auc_score(y, L.s_svc), "direction": "higher = TP",
                "n_values": None, "fill_TP": 1.0, "fill_FP": 1.0, "source": "reference"})
    for c in ["V_N_SERVICES", "V_N_NON_EM"]:
        au = roc_auc_score(y, L[c])
        res.append({"column": c, "kind": "numeric (visit)", "AUC": max(au, 1 - au), "direction": "higher = TP" if au >= .5 else "lower = TP",
                    "n_values": int(L[c].nunique()), "fill_TP": 1.0, "fill_FP": 1.0, "source": "derived"})
        values[c] = L[c]
    for c in feats_model:
        res.append({"column": c, "kind": "categorical", "AUC": cat_auc(L[c], y), "direction": "", "n_values": int(L[c].nunique()),
                    "fill_TP": float(key(L[c])[y == 1].ne("").mean()), "fill_FP": float(key(L[c])[y == 0].ne("").mean()), "source": "model feature"})
    for i in range(0, len(keep), 20):
        tab = dset.to_table(columns=keep[i:i + 20]).take(idx).to_pandas()
        for c in tab.columns:
            v = tab[c]
            k = key(v)
            filled = k.ne("")
            num = pd.to_numeric(v, errors="coerce") if not pd.api.types.is_bool_dtype(v) else v.astype(float)
            nun = k[filled].nunique()
            if c in feats_model or k.nunique() <= 1:          # model features are already listed; constant columns say nothing
                continue
            is_id = bool(re.search(r"_ID$|NPI|GUID|^ID_|_NUM$|NUMBER$", c.upper()))   # codes, not quantities
            if not is_id and num.notna().mean() > 0.95 * filled.mean() and nun > 20:
                au = roc_auc_score(y, num.fillna(num.median()))
                kind, auc, d = "numeric", max(au, 1 - au), ("higher = TP" if au >= .5 else "lower = TP")
            else:
                kind, auc, d = "categorical", cat_auc(v, y), ""
            res.append({"column": c, "kind": kind, "AUC": auc, "direction": d, "n_values": int(nun),
                        "fill_TP": float(filled[y == 1].mean()), "fill_FP": float(filled[y == 0].mean()), "source": "data file"})
            if nun <= PRINT_VALUES_MAX or kind == "numeric" or (is_id and nun <= 400):
                values[c] = v
        print(f"  scanned {min(i + 20, len(keep))}/{len(keep)} columns")

    R = pd.DataFrame(res)
    R["fill_gap"] = (R.fill_TP - R.fill_FP).round(3)
    R = R.sort_values("AUC", ascending=False)
    R[["AUC", "fill_TP", "fill_FP"]] = R[["AUC", "fill_TP", "fill_FP"]].round(3)
    R.to_csv(f"{a.out}/1_column_scan.csv", index=False)
    print("\n== 1. columns ranked by how well each alone separates E/M TP from FP (AUC 0.5 = nothing)\n",
          R.head(40).to_string(index=False))

    print("\n== 2. value detail for the top separating columns (low-cardinality or numeric only)")
    shown = 0
    for c in R[R.source.isin(["data file", "derived"])].column:
        if c not in values or shown >= 10:
            continue
        v = values[c]
        if pd.api.types.is_numeric_dtype(v) and v.nunique() > PRINT_VALUES_MAX and not re.search(r"_ID$|NPI|GUID", c.upper()):
            b = pd.qcut(v.rank(method="first"), 5, labels=[f"q{i}" for i in range(1, 6)])
            g = pd.DataFrame({"bin": b, "v": v, "y": y}).groupby("bin", observed=True).agg(lo=("v", "min"), hi=("v", "max"), TP=("y", "sum"), n=("y", "size"))
        else:
            g = pd.DataFrame({"val": key(v).replace("", "(missing)"), "y": y}).groupby("val").y.agg(TP="sum", n="size")
            g = g[g.n >= 20].sort_values("n", ascending=False).head(15)
        g["FP"] = g.n - g.TP
        g["PA_share"] = (g.TP / g.n).round(3)
        g.to_csv(f"{a.out}/2_values_{re.sub(r'[^A-Za-z0-9_]', '_', c)}.csv")
        print(f"\n-- {c}\n{g.drop(columns='n').to_string()}")
        shown += 1
    print(f"\noutputs -> {a.out}/  (base PA share among flagged E/M here: {y.mean():.3f})")
    os._exit(0)


if __name__ == "__main__":
    main()
