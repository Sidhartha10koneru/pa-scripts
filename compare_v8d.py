"""
compare_v8d.py - compare models trained on different label columns (v8 draft) fairly.

Problem: each run is tested on ITS OWN labels, so the runs are not scored on the same target. A passenger model is
"wrong" on passenger services under the original labels by design, and V7 models gain extra TRUE services.

So every model is scored on the SAME services under EVERY label set (a model x label matrix):
  services  = CSN + CPT present in every run's test set, quality codes left out; label = max, score = max over lines
  label set = the y of each run's own test_preds (GROUP_LABEL run = original labels)
  read a column: which model ranks best if THAT label set is the truth
  read the diagonal vs the GROUP_LABEL row: does training on the new labels beat the base model on the new target?
Plus:
  agree   services where ALL label sets agree (labels not in dispute) - a clean, label-neutral check
  visit   the same matrix at visit level (CSN: PA if any service PA; score = max)
  pax     with --v8-path: share of passenger / visit-level-auth services each model puts above its 80%-recall
          cut-off (original labels) - shows the intended effect (passengers ranked lower)

Run:
  python compare_v8d.py base=exp_v8d_grp_s42 l7spax=exp_v8d_l7spax_s42 l6pax=exp_v8d_l6pax_s42 \
         l7pax=exp_v8d_l7pax_s42 --base base --out v8d_compare \
         --v8-path "s3://cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v8_draft/ai_prior_auth_09282026_v3_{year}_std.parquet"
Outputs: v8d_compare/matrix_service.csv, matrix_visit.csv, agree.csv, pax.csv, label_overlap.csv
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

QUALITY = r"^\d{4}F$"
TRUE_SET = {"TRUE", "1", "1.0"}


def services(path):
    t = pd.read_parquet(path)
    t = t.loc[:, ~t.columns.duplicated()]
    p = [c for c in t.columns if c.startswith("p_")]
    if len(p) != 1:
        raise SystemExit(f"{path}: expected one score column p_*, found {p}")
    t["CSN"] = t.CSN.astype(str).str.replace(r"\.0$", "", regex=True)
    t["FINAL_CPT"] = t.FINAL_CPT.astype(str).str.strip().str.upper()
    t = t[~t.FINAL_CPT.str.fullmatch(QUALITY)]
    return t.groupby(["CSN", "FINAL_CPT"], observed=True).agg(y=("y", "max"), score=(p[0], "max"))


def metrics(y, s):
    y, s = np.asarray(y), np.asarray(s)
    cut = np.quantile(s[y == 1], 0.2)
    f = s >= cut
    tp, fp = int((f & (y == 1)).sum()), int((f & (y == 0)).sum())
    return {"n": len(y), "PA": int(y.sum()), "AUC_PR": round(average_precision_score(y, s), 4),
            "AUC_ROC": round(roc_auc_score(y, s), 4), "FP@80": fp, "prec@80": round(tp / max(tp + fp, 1), 4)}


def matrix(S, names, level):
    rows = []
    for lab in names:                      # label set = truth
        for m in names:                    # model scored
            if level == "visit":
                v = S.groupby(level="CSN").agg(y=(f"y_{lab}", "max"), s=(f"s_{m}", "max"))
                r = metrics(v.y, v.s)
            else:
                r = metrics(S[f"y_{lab}"], S[f"s_{m}"])
            rows.append({"labels": lab, "model": m, **r})
    return pd.DataFrame(rows)


def show(df, base, title):
    print(f"\n== {title}")
    for k in ["AUC_PR", "FP@80", "prec@80"]:
        p = df.pivot(index="model", columns="labels", values=k)
        print(f"  {k} (rows = model, columns = labels used as truth)\n" + p.to_string())
    for lab in df["labels"].unique():
        d = df[df["labels"] == lab].set_index("model")
        if base in d.index:
            best = d.AUC_PR.idxmax()
            print(f"  labels={lab}: n={int(d.n.iloc[0]):,} PA={int(d.PA.iloc[0]):,} | best AUC-PR {best} "
                  f"{d.AUC_PR[best]:.4f} vs {base} {d.AUC_PR[base]:.4f} ({d.AUC_PR[best] - d.AUC_PR[base]:+.4f})")


def pax_flags(path, csns):
    import pyarrow.dataset as ds
    import pyarrow.fs as pafs
    out = []
    for y in (2025, 2026):
        p = path.format(year=y)
        fs, p = (pafs.S3FileSystem(), p.replace("s3://", "")) if p.startswith("s3://") else (pafs.LocalFileSystem(), p)
        t = ds.dataset(p, filesystem=fs, format="parquet").to_table(
            columns=["CSN", "FINAL_CPT", "IS_PASSENGER", "VISIT_LEVEL_AUTH"]).to_pandas()
        t["CSN"] = t.CSN.astype(str).str.replace(r"\.0$", "", regex=True)
        t = t[t.CSN.isin(csns)]
        t["FINAL_CPT"] = t.FINAL_CPT.astype(str).str.strip().str.upper()
        out.append(t)
    t = pd.concat(out)
    return t.groupby(["CSN", "FINAL_CPT"]).agg(pax=("IS_PASSENGER", "max"), vla=("VISIT_LEVEL_AUTH", "max"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="name=run_folder ...")
    ap.add_argument("--base", required=True, help="name of the GROUP_LABEL run (original labels)")
    ap.add_argument("--out", default="v8d_compare")
    ap.add_argument("--v8-path", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pd.set_option("display.width", 250)
    runs = dict(r.split("=", 1) for r in a.runs)
    names = list(runs)
    if a.base not in runs:
        raise SystemExit("--base must be one of the run names")

    parts = {}
    for n, f in runs.items():
        s = services(f"{f}/test_preds.parquet")
        parts[n] = s.rename(columns={"y": f"y_{n}", "score": f"s_{n}"})
        print(f"{n:10s} {f}: services {len(s):,} PA {int(s.y.sum()):,} ({s.y.mean():.2%})")
    S = pd.concat(parts.values(), axis=1, join="inner")
    print(f"common services (in every run's test set): {len(S):,}")

    # how far the label sets differ on the common services
    ov = pd.DataFrame([{"labels": n, "PA": int(S[f"y_{n}"].sum()),
                        "TRUE_where_base_FALSE": int(((S[f"y_{n}"] == 1) & (S[f"y_{a.base}"] == 0)).sum()),
                        "FALSE_where_base_TRUE": int(((S[f"y_{n}"] == 0) & (S[f"y_{a.base}"] == 1)).sum())}
                       for n in names])
    ov.to_csv(f"{a.out}/label_overlap.csv", index=False)
    print("\n== labels on the common services vs the original labels\n" + ov.to_string(index=False))

    ms = matrix(S, names, "service"); ms.to_csv(f"{a.out}/matrix_service.csv", index=False)
    show(ms, a.base, "SERVICE level: every model x every label set, same services")
    mv = matrix(S, names, "visit"); mv.to_csv(f"{a.out}/matrix_visit.csv", index=False)
    show(mv, a.base, "VISIT level")

    ycols = [f"y_{n}" for n in names]
    agree = S[S[ycols].nunique(axis=1) == 1]
    ag = pd.DataFrame([{"model": m, **metrics(agree[f"y_{a.base}"], agree[f"s_{m}"])} for m in names])
    ag.to_csv(f"{a.out}/agree.csv", index=False)
    print(f"\n== services where ALL label sets agree ({len(agree):,}; {len(S) - len(agree):,} disputed left out)\n"
          + ag.to_string(index=False))

    if a.v8_path:
        f = pax_flags(a.v8_path, set(S.index.get_level_values("CSN")))
        P = S.join(f, how="left").fillna({"pax": 0, "vla": 0})
        rows = []
        for m in names:
            y, s = P[f"y_{a.base}"].to_numpy(), P[f"s_{m}"].to_numpy()
            cut = np.quantile(s[y == 1], 0.2)
            for g, mask in [("passenger", P.pax.eq(1)), ("visit_level_auth", P.vla.eq(1)),
                            ("other PA (original)", P.pax.eq(0) & P.vla.eq(0) & (y == 1))]:
                rows.append({"model": m, "group": g, "services": int(mask.sum()),
                             "above_80pct_cutoff": round(float((s[mask.to_numpy()] >= cut).mean()), 4) if mask.any() else None})
        px = pd.DataFrame(rows); px.to_csv(f"{a.out}/pax.csv", index=False)
        print("\n== share of services above each model's 80%-recall cut-off (original labels)\n"
              + px.pivot(index="model", columns="group", values="above_80pct_cutoff").to_string())
        print("   passenger: lower = intended effect; other PA: should NOT drop (= real PA still caught)")
    print(f"\noutputs -> {a.out}/")


if __name__ == "__main__":
    main()
