"""
precision_ceiling.py - how high can precision go with ONLY the 5 model fields (CPT, modifier, POS, payer, plan)?

Compares the trained model with "perfect" scorers that know the true PA rate of every field combination:

  model            the run's own scores
  ceiling (oracle) rank services by the PA rate of their combination IN THE TEST SET ITSELF.
                   No model using only these 5 fields can beat this on this test set. Optimistic: a combination
                   seen once is trivially 0% or 100%, so it can look better than anything achievable.
  ceiling (2-fold) the same, but the rate of a combination is learned on one half of the test visits and applied
                   to the other half (unseen combination -> rate of its CPT -> overall rate).
                   A realistic ceiling: perfect, up-to-date knowledge of every combination, no memorisation.
  lookup (train)   optional (--with-history): rank by the combination's PA rate in TRAINING data.
                   The simplest "model" possible; uses fn_buckets.py's cached training history.

A service = CSN + CPT (as everywhere else). Its combination = CPT + the set of modifier / POS / payer|plan of all
its lines. Score of a service = max over its lines.

If the realistic ceiling at 80% recall is far below 90% precision, 90% needs NEW information (diagnosis, prior
auths, provider, payer rules), not a better model. If the ceiling is far above the model, the model is the issue.

Needs fn_buckets.py in the same folder (pandas, numpy, pyarrow). Run from the experiments folder:
  python precision_ceiling.py --run exp_v8d_l6pax_s42 \
      --frozen frozen/test_v1_2026-05_2026-07 --labels-from exp_v8d_l6pax_s42 [--with-history]
Output: <run>/precision_ceiling/{metrics.csv, pa_by_combination_rate.csv} + printed tables.
"""
import argparse
import os

import numpy as np
import pandas as pd

import fn_buckets as fb

RECALLS = (0.70, 0.75, 0.80, 0.85, 0.90)
PRECISIONS = (0.50, 0.60, 0.70, 0.80, 0.90)
LINE_CELL = ["FINAL_MODIFIERS", "PLACE_OF_SERVICE_CODE", "PAYOR_PLAN_CLEAN"]


# ---------------------------------------------------------------------------------------------------- PR maths
def curve(y, score):
    """PR curve with ties handled fairly: services with the same score are taken together, and inside a tie
    group recall/precision move linearly (= expected value of picking at random within the group)."""
    d = pd.DataFrame({"y": np.asarray(y, int), "s": np.asarray(score, float)})
    g = d.groupby("s", sort=True).y.agg(["sum", "size"]).iloc[::-1]          # highest score first
    tp = np.concatenate([[0], g["sum"].cumsum().to_numpy()])
    fp = np.concatenate([[0], (g["size"] - g["sum"]).cumsum().to_numpy()])
    return tp.astype(float), fp.astype(float), float(d.y.sum())


def at_recall(c, r):
    tp, fp, P = c
    want = r * P
    i = int(np.searchsorted(tp, want, side="left"))
    i = min(max(i, 1), len(tp) - 1)
    f = (want - tp[i - 1]) / max(tp[i] - tp[i - 1], 1e-12)
    FP = fp[i - 1] + f * (fp[i] - fp[i - 1])
    return {"precision": want / max(want + FP, 1e-12), "TP": want, "FP": FP, "FN": P - want}


def recall_at_precision(c, p, grid=np.linspace(0.001, 1, 1000)):
    ok = [r for r in grid if at_recall(c, r)["precision"] >= p]
    return max(ok) if ok else 0.0


def auc_pr(c):
    tp, fp, P = c
    rec, prec = tp / P, np.divide(tp, tp + fp, out=np.ones_like(tp), where=(tp + fp) > 0)
    return float(np.sum(np.diff(rec) * prec[1:]))                            # average precision


# ---------------------------------------------------------------------------------------------------- combinations
def service_combinations(preds, rename=None):
    """CSN + CPT -> combination = CPT + sorted set of modifier/POS/payer|plan over its lines"""
    t = fb.read_any(preds, rename)
    if "PAYOR_PLAN_CLEAN" not in t and {"PAYOR", "PLAN"} <= set(t.columns):
        t["PAYOR_PLAN_CLEAN"] = fb.clean_name(t.PAYOR) + " | " + fb.clean_name(t.PLAN)
    t["CSN"] = t.CSN.astype(str).str.replace(r"\.0$", "", regex=True)
    for c in fb.KEY:
        t[c] = t[c].astype(object).fillna("").astype(str).str.strip()
    t["line_cell"] = t[LINE_CELL].agg("/".join, axis=1)
    sig = (t.drop_duplicates(["CSN", "FINAL_CPT", "line_cell"])
           .groupby(["CSN", "FINAL_CPT"]).line_cell.agg(lambda x: " + ".join(sorted(x))).rename("combo").reset_index())
    sig["combo"] = sig.FINAL_CPT + " :: " + sig.combo
    return t, sig


def oracle_rate(s):
    return s.groupby("combo").y.transform("mean")


def two_fold_rate(s, seed=0):
    """combination rate learned on the other half of the visits; unseen -> CPT rate -> overall rate"""
    rng = np.random.default_rng(seed)
    visits = s.CSN.unique()
    half = pd.Series(rng.integers(0, 2, len(visits)), index=visits)
    fold = s.CSN.map(half).to_numpy()
    out = np.empty(len(s))
    for k in (0, 1):
        tr, te = s[fold != k], s[fold == k]
        combo, cpt, overall = tr.groupby("combo").y.mean(), tr.groupby("FINAL_CPT").y.mean(), tr.y.mean()
        r = te.combo.map(combo)
        r = r.fillna(te.FINAL_CPT.map(cpt)).fillna(overall)
        out[fold == k] = r.to_numpy()
    return out


def train_lookup(lines, s, a):
    """max over a service's lines of the training PA rate of the line's cell (unseen -> CPT rate -> 0)"""
    st = fb.settings(a)
    cell, cpt = fb.cell_history(st)
    l = lines[["CSN"] + fb.KEY].merge(cell, on=fb.KEY, how="left").merge(cpt, on="FINAL_CPT", how="left")
    l["r"] = l.train_pa_rate.where(l.train_n >= a.min_n).fillna(l.train_cpt_pa_rate).fillna(0)
    r = l.groupby(["CSN", "FINAL_CPT"]).r.max().rename("lookup").reset_index()
    return s.merge(r, on=["CSN", "FINAL_CPT"], how="left").lookup.fillna(0).to_numpy()


# ---------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--preds", default=None, help="default <run>/test_preds.parquet")
    ap.add_argument("--score-col", default=None)
    ap.add_argument("--frozen", default=None)
    ap.add_argument("--labels-from", default=None)
    ap.add_argument("--with-history", action="store_true", help="add the training-lookup scorer (needs S3 or cache)")
    ap.add_argument("--min-n", type=int, default=20, help="lookup: fewer training lines -> use the CPT rate")
    ap.add_argument("--out", default=None)
    for f in ("--base-path", "--label", "--quality-mode", "--test-start"):          # only for --with-history
        ap.add_argument(f, default=None)
    ap.add_argument("--years", nargs="+", type=int, default=None)
    ap.add_argument("--include-secondary", default=None, type=lambda v: v.lower() in ("1", "true", "yes"))
    a = ap.parse_args()
    out = a.out or f"{a.run}/precision_ceiling"
    os.makedirs(out, exist_ok=True)
    preds = fb.predictions_path(a)

    s, src = fb.test_services(preds, a.frozen, a.labels_from, a.score_col)
    lines, sig = service_combinations(preds)
    s = s.merge(sig, on=["CSN", "FINAL_CPT"], how="left")
    n_combo = s.combo.nunique()
    print(f"[combos] {n_combo:,} field combinations in test | "
          f"{(s.groupby('combo').size() == 1).sum():,} seen only once (these make the oracle optimistic)")

    scorers = {"model": s.score.to_numpy(),
               "ceiling (2-fold)": two_fold_rate(s),
               "ceiling (oracle)": oracle_rate(s).to_numpy()}
    if a.with_history:
        scorers = {"model": scorers["model"], "lookup (train)": train_lookup(lines, s, a), **{k: v for k, v in scorers.items() if k != "model"}}

    rows = []
    for name, sc in scorers.items():
        c = curve(s.y, sc)
        row = {"scorer": name, "AUC-PR": auc_pr(c)}
        for r in RECALLS:
            m = at_recall(c, r)
            row[f"precision @ recall {r:.0%}"] = m["precision"]
            row[f"FP @ recall {r:.0%}"] = round(m["FP"])
        for p in PRECISIONS:
            row[f"recall @ precision {p:.0%}"] = recall_at_precision(c, p)
        rows.append(row)
    R = pd.DataFrame(rows)
    R.insert(1, "test", src)
    R.to_csv(f"{out}/metrics.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    P = int(s.y.sum())
    print(f"\n== {a.run} | {len(s):,} services, {P:,} PA ({P / len(s):.2%}) | {src}")
    print("\nPrecision at a given recall (FP = false alarms):")
    cols = ["scorer"] + [c for r in RECALLS for c in (f"precision @ recall {r:.0%}", f"FP @ recall {r:.0%}")]
    print(R[cols].round(3).to_string(index=False))
    print("\nRecall reachable at a given precision:")
    print(R[["scorer", "AUC-PR"] + [f"recall @ precision {p:.0%}" for p in PRECISIONS]].round(3).to_string(index=False))

    # where the PA services sit: combinations that are always / sometimes / rarely PA (test, 2-fold rate)
    s["rate"] = scorers["ceiling (2-fold)"]
    bands = pd.cut(s.rate, [-0.01, 0.10, 0.50, 0.90, 1.0], labels=["<10%", "10-50%", "50-90%", ">=90%"])
    B = s.groupby(bands, observed=False).agg(services=("y", "size"), PA=("y", "sum"))
    B["non-PA"] = B.services - B.PA
    B["share of all PA"] = B.PA / P
    B.index.name = "combination PA rate (2-fold)"
    B.to_csv(f"{out}/pa_by_combination_rate.csv")
    print("\nWhere the PA services sit, by how often their combination is PA:")
    print(B.round(3).to_string())
    print(f"\nwritten: {out}/metrics.csv, {out}/pa_by_combination_rate.csv")


if __name__ == "__main__":
    main()


python precision_ceiling.py --run exp_v8d_l6pax_s42 \
    --frozen frozen/test_v1_2026-05_2026-07 --labels-from exp_v8d_l6pax_s42

