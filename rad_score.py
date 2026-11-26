"""
rad_score.py - score several runs on radiology test services, under several answer keys, two scoring rules.

For each run (name=folder) it keeps radiology test lines (CPT range), builds services (CSN + CPT) and computes:
  answer keys   each --labels column, read per line from the data file by SRC_ROW; service = PA if any line TRUE,
                not PA if no TRUE and at least one FALSE; services with neither are left out of that key
  scoring       max      = highest line score (today's rule)
                primary  = highest score among filing-order-1 lines; highest of any line when there is none
  recall        cut-off chosen per run x key x scoring at each --recall (0.85 = our headline, 0.8948 = Sumit's)
Only services present in EVERY run are scored, so all rows compare the same services (the count is printed).

  python rad_score.py core=exp_feat_core_s42 rad_v6pax=exp_rad_v6_pax_s42 rad_v7strict=exp_rad_v7_strict_s42 \
      --out rad_scores.csv
"""
import argparse
import os

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.fs as pafs

DATA = "s3://cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v8_draft/ai_prior_auth_09282026_v3_2026_std.parquet"
TRUE_V, FALSE_V = {"TRUE", "1", "1.0"}, {"FALSE", "0", "0.0"}


def key(s):
    return s.astype(object).where(s.notna(), None).map(lambda v: "" if v is None else str(v)) \
        .str.strip().str.replace(r"^(\d+)\.0$", r"\1", regex=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="name=folder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--test-year", type=int, default=2026)
    ap.add_argument("--labels", nargs="+", default=["LABEL_V6_PAX", "LABEL_V7_STRICT"])
    ap.add_argument("--recall", nargs="+", type=float, default=[0.85, 0.8948])
    ap.add_argument("--cpt-from", type=int, default=70010)
    ap.add_argument("--cpt-to", type=int, default=79999)
    a = ap.parse_args()
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)

    fs, p = (pafs.S3FileSystem(), a.data.replace("s3://", "")) if a.data.startswith("s3://") else (pafs.LocalFileSystem(), os.path.abspath(a.data))
    dset = ds.dataset(p, filesystem=fs, format="parquet")
    labs = [l for l in a.labels if l in dset.schema.names]
    lab_tab = dset.to_table(columns=["CSN"] + labs).to_pandas()          # read once, rows picked by SRC_ROW per run
    lab_tab["k_csn"] = key(lab_tab.CSN)
    for l in labs:
        v = key(lab_tab[l]).str.upper()
        lab_tab[l] = np.where(v.isin(TRUE_V), 2, np.where(v.isin(FALSE_V), 1, 0))

    svc = {}
    for spec in a.runs:
        name, folder = spec.split("=", 1)
        t = pd.read_parquet(f"{folder}/test_preds.parquet")
        t = t.loc[:, ~t.columns.duplicated()]
        pc = [c for c in t.columns if c.startswith("p_")][0]
        t["k_csn"], t["k_cpt"] = key(t.CSN), key(t.FINAL_CPT).str.upper()
        n = pd.to_numeric(t.k_cpt.where(t.k_cpt.str.fullmatch(r"\d{5}")), errors="coerce")
        t = t[n.between(a.cpt_from, a.cpt_to)].copy()
        rows = pd.to_numeric(t.SRC_ROW, errors="coerce")
        if "SRC_YEAR" in t:
            rows = rows.where(pd.to_numeric(t.SRC_YEAR, errors="coerce") == a.test_year)
        ok = rows.notna() & rows.between(0, len(lab_tab) - 1)
        r = rows[ok].astype(int).values
        match = (lab_tab.k_csn.values[r] == t.k_csn[ok].values).mean()
        print(f"{name}: radiology test lines {len(t):,} | labels found by SRC_ROW {ok.mean():.1%} | CSN match {match:.1%}")
        if match < 0.99:
            raise SystemExit(f"{name}: SRC_ROW does not line up with --data")
        for l in labs:
            t[l] = 0
            t.loc[ok, l] = lab_tab[l].values[r]
        t["prim"] = key(t.FILING_ORDER_C).eq("1") if "FILING_ORDER_C" in t else True
        g = t.groupby(["k_csn", "k_cpt"])
        S = g.agg(s_max=(pc, "max"), has_prim=("prim", "max"), **{l: (l, "max") for l in labs})
        S["s_primary"] = t[t.prim].groupby(["k_csn", "k_cpt"])[pc].max().reindex(S.index)
        S["s_primary"] = S.s_primary.where(S.has_prim, S.s_max)
        svc[name] = S
    common = None
    for S in svc.values():
        common = S.index if common is None else common.intersection(S.index)
    print(f"\nservices scored (present in every run): {len(common):,}")

    out = []
    for name, S in svc.items():
        S = S.loc[common]
        for l in labs:
            kn = S[l] > 0
            y = (S[l] == 2)[kn].values
            for rule in ["max", "primary"]:
                s = S[f"s_{rule}"][kn].values
                for rc in a.recall:
                    cut = float(np.quantile(s[y], 1 - rc))
                    f = s >= cut
                    tp, fp = int((f & y).sum()), int((f & ~y).sum())
                    fn, tn = int((~f & y).sum()), int((~f & ~y).sum())
                    out.append({"run": name, "answer_key": l, "scoring": rule, "target_recall": rc, "cut": round(cut, 4),
                                "services": int(kn.sum()), "PA": int(y.sum()), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                                "precision": round(tp / max(tp + fp, 1), 4), "recall": round(tp / max(tp + fn, 1), 4)})
    R = pd.DataFrame(out).sort_values(["answer_key", "target_recall", "scoring", "precision"], ascending=[True, True, True, False])
    R.to_csv(a.out, index=False)
    print(R.to_string(index=False))
    print(f"\n-> {a.out}  (Sumit's radiology reference: 69.4% precision at 89.5% recall, his labels and service set)")
    os._exit(0)


if __name__ == "__main__":
    main()
