"""
em_check.py - why does the model raise false alarms on E/M (office visit) services?  Counts only.

At the cut-off for --recall on ALL test services (same cut-off as the headline), for E/M services (CPT 99202-99499):
  1_by_cpt          TP/FP/FN/TN + precision per E/M code (new vs established vs consult ...)
  2_by_plan         same, per payor | plan (top 20 by FP)
  3_by_specialty    same, per SPECIALTY (top 20 by FP)
  4_visit_context   E/M FP / TN split by: other PA service on the same visit (CSN)?  E/M line relabelled as a passenger
                    (original LABEL_V6 TRUE -> LABEL_V6_PAX FALSE)?  IS_PASSENGER?
  5_train_rate      training PA rate of the same payor | plan | CPT (dos < --train-end) for TP / FP / FN / TN
                    -> FP with a high training rate = the model is following history (labels / policy), not inventing it
  6_monthly         E/M PA rate by month for the top-5 FP plans, train + test months (drift check)
No patient IDs, no auth values. Writes <out>/*.csv and prints them.

  nohup python -u em_check.py --run exp_feat_core_s42 --out em_core > run_em_check.log 2>&1 &
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.fs as pafs

V8 = "s3://cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v8_draft/ai_prior_auth_09282026_v3_{year}_std.parquet"
TRUE_SET = {"TRUE", "1", "1.0"}
WANT = ["CSN", "FINAL_CPT", "SERVICEDATE", "PAYOR", "PLAN", "SPECIALTY", "PLACE_OF_SERVICE_CODE",
        "LABEL_V6", "LABEL_V6_PAX", "IS_PASSENGER"]


def key(s):
    return s.astype(object).fillna("").astype(str).str.replace(r"\.0$", "", regex=True).str.strip().str.upper()


def clean(s):
    return key(s).str.replace(r"\bAS OF\b.*$", "", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()


def is_em(cpt):
    n = pd.to_numeric(cpt.where(cpt.str.fullmatch(r"\d{5}")), errors="coerce")
    return n.between(99202, 99499)


def load(year, path):
    p = path.format(year=year)
    fs, p = (pafs.S3FileSystem(), p.replace("s3://", "")) if p.startswith("s3://") else (pafs.LocalFileSystem(), os.path.abspath(p))
    d = ds.dataset(p, filesystem=fs, format="parquet")
    cols = [c for c in WANT if c in d.schema.names]
    return d.to_table(columns=cols).to_pandas(), cols


def cm_table(x, by, top=None):
    c = x.groupby(by, observed=True).cm.value_counts().unstack(fill_value=0).reindex(columns=["TP", "FP", "FN", "TN"], fill_value=0)
    c["services"] = c.sum(axis=1)
    c["precision"] = (c.TP / (c.TP + c.FP).replace(0, np.nan)).round(3)
    c["recall"] = (c.TP / (c.TP + c.FN).replace(0, np.nan)).round(3)
    c["FP_share"] = (c.FP / max(int(c.FP.sum()), 1)).round(3)
    c = c.sort_values("FP", ascending=False)
    return c.head(top) if top else c


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-path", default=V8)
    ap.add_argument("--years", nargs="+", type=int, default=[2022, 2023, 2024, 2025, 2026])
    ap.add_argument("--train-end", default="2026-02-01")
    ap.add_argument("--recall", type=float, default=0.85)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)

    # test services + cut-off (all services, like the headline)
    t = pd.read_parquet(f"{a.run}/test_preds.parquet")
    t = t.loc[:, ~t.columns.duplicated()]
    p = [c for c in t.columns if c.startswith("p_")][0]
    t["k_csn"], t["k_cpt"] = key(t.CSN), key(t.FINAL_CPT)
    S = t.groupby(["k_csn", "k_cpt"]).agg(y=("y", "max"), s=(p, "max")).reset_index()
    S["y"] = S.y.astype(bool)
    cut = float(np.quantile(S.s[S.y], 1 - a.recall))
    S["cm"] = np.select([S.y & (S.s >= cut), ~S.y & (S.s >= cut), S.y], ["TP", "FP", "FN"], "TN")
    S["em"] = is_em(S.k_cpt)
    visit_pa = (S.y & ~S.em).groupby(S.k_csn).any()                      # visit has a non-E/M PA service
    print(f"cut-off {cut:.6f} | E/M services {int(S.em.sum()):,} | E/M FP {int(((S.cm == 'FP') & S.em).sum()):,} of {int((S.cm == 'FP').sum()):,}")
    test_csn = set(S.k_csn)

    # data file: E/M lines only, all years
    parts, train_rows = [], []
    for y in a.years:
        t0 = time.time()
        d, cols = load(y, a.base_path)
        d["k_cpt"] = key(d.FINAL_CPT)
        d = d[is_em(d.k_cpt)].copy()
        d["k_csn"] = key(d.CSN)
        d["dos"] = pd.to_datetime(d.SERVICEDATE, errors="coerce")
        d["plan"] = clean(d.PAYOR) + " | " + (clean(d.PLAN) if "PLAN" in d else "")
        d["pax_label"] = key(d.LABEL_V6_PAX).isin(TRUE_SET) if "LABEL_V6_PAX" in d else False
        d["orig_label"] = key(d.LABEL_V6).isin(TRUE_SET) if "LABEL_V6" in d else d.pax_label
        d["is_pax"] = pd.to_numeric(d.IS_PASSENGER, errors="coerce").fillna(0).eq(1) if "IS_PASSENGER" in d else False
        d["spec"] = key(d.SPECIALTY) if "SPECIALTY" in d else "?"
        d["pos"] = key(d.PLACE_OF_SERVICE_CODE) if "PLACE_OF_SERVICE_CODE" in d else "?"
        keep = ["k_csn", "k_cpt", "dos", "plan", "spec", "pos", "pax_label", "orig_label", "is_pax"]
        train_rows.append(d.loc[d.dos < a.train_end, keep])
        parts.append(d.loc[d.k_csn.isin(test_csn), keep])
        print(f"[{y}] E/M lines {len(d):,} | {time.time() - t0:.0f}s | columns missing: {sorted(set(WANT) - set(cols)) or 'none'}")
        del d
    A = pd.concat(parts, ignore_index=True)
    T = pd.concat(train_rows, ignore_index=True)

    # one row per test E/M service
    E = S[S.em].merge(A.groupby(["k_csn", "k_cpt"]).agg(plan=("plan", "first"), spec=("spec", "first"), pos=("pos", "first"),
                                                         orig=("orig_label", "max"), is_pax=("is_pax", "max")).reset_index(),
                      on=["k_csn", "k_cpt"], how="left")
    E["other_PA_on_visit"] = E.k_csn.map(visit_pa).fillna(False).astype(bool)
    E["relabelled_passenger"] = E.orig.fillna(False).astype(bool) & ~E.y
    E["is_pax"] = E.is_pax.fillna(False).astype(bool)
    print(f"test E/M services matched to data file: {E.plan.notna().mean():.1%}")

    out = {}
    out["1_by_cpt"] = cm_table(E, "k_cpt", 25)
    out["2_by_plan"] = cm_table(E, "plan", 20)
    out["3_by_specialty"] = cm_table(E, "spec", 20)
    E["context"] = np.select([E.relabelled_passenger, E.other_PA_on_visit], ["relabelled passenger (orig TRUE)", "other PA service on visit"],
                             "E/M alone or with non-PA services")
    out["4_visit_context"] = cm_table(E, "context")

    # training PA rate of the same plan | CPT (service level, PA-first dedup)
    Ts = T.groupby(["k_csn", "k_cpt"]).agg(plan=("plan", "first"), y=("pax_label", "max")).reset_index()
    rate = Ts.groupby(["plan", "k_cpt"]).y.agg(n_train="size", rate_train="mean").reset_index()
    E = E.merge(rate, on=["plan", "k_cpt"], how="left")
    E["train_rate_bin"] = pd.cut(E.rate_train, [-0.01, 0.05, 0.2, 0.5, 0.8, 1.0],
                                 labels=["<5%", "5-20%", "20-50%", "50-80%", ">80%"]).astype(object).where(E.n_train >= 20, "unseen/<20")
    out["5_train_rate"] = cm_table(E, "train_rate_bin").sort_index()

    # monthly E/M PA rate, top-5 FP plans
    top5 = out["2_by_plan"].index[:5]
    M = pd.concat([T, A])
    M = M[M.plan.isin(top5)].groupby(["k_csn", "k_cpt"]).agg(plan=("plan", "first"), dos=("dos", "min"), y=("pax_label", "max")).reset_index()
    M["month"] = M.dos.dt.to_period("M").astype(str)
    mo = M.groupby(["plan", "month"]).y.agg(services="size", pa_rate="mean").round(4).reset_index()
    out["6_monthly"] = mo.pivot(index="month", columns="plan", values="pa_rate")

    for k, v in out.items():
        v.to_csv(f"{a.out}/{k}.csv")
        print(f"\n== {k}\n{v.to_string()}")
    print(f"\noutputs -> {a.out}/ (counts and rates only)")
    os._exit(0)


if __name__ == "__main__":
    main()


# nohup python -u em_check.py --run exp_feat_core_s42 --out em_core > run_em_check.log 2>&1 &
# tail -f run_em_check.log
