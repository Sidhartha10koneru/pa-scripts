"""
hist_check.py - does PA history separate PA from false alarms? Checked BEFORE building the new data folder.

Computes the PA-history columns in memory (nothing written to S3) for the test window only, joins them to the core
model's test scores, and prints at service level (CSN + CPT, quality codes out):
  1. PA rate with vs without history, per CPT family     -> is there signal at all?
  2. share of TP / FP / FN / TN with history (85% recall) -> would it move false alarms?
  3. per family: false alarms WITHOUT history vs PA caught WITH history -> where it could help
Needs build_pa_history.py in the same folder.

  python hist_check.py --run exp_feat_core_s42
"""
import argparse
import os

import numpy as np
import pandas as pd

import build_pa_history as b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="exp_feat_core_s42")
    ap.add_argument("--recall", type=float, default=0.85)
    ap.add_argument("--lag-days", type=int, default=0)
    a = ap.parse_args()
    lo = 1 + a.lag_days

    # PA events from all years (same rules as the build)
    parts = []
    for y in [2022, 2023, 2024, 2025, 2026]:
        fs, p = b.fs_path(b.V8D.format(year=y))
        d = pd.read_parquet(p, columns=["PAT_ID", "FINAL_CPT", "SERVICEDATE", "LABEL_V6_PAX"], filesystem=fs)
        cpt = b.norm(d.FINAL_CPT)
        m = b.norm(d.LABEL_V6_PAX).isin(b.TRUE_SET) & ~cpt.str.fullmatch(b.QUALITY)
        e = pd.DataFrame({"pat": b.norm(d.PAT_ID[m]), "cpt": cpt[m], "day": b.days(d.SERVICEDATE[m])})
        parts.append(e[(e.day >= 0) & ~e.pat.isin(["", "NAN", "NONE"])].drop_duplicates())
    E = pd.concat(parts, ignore_index=True).drop_duplicates()
    fam_of = {c: b.family(c) for c in E.cpt.unique()}
    E["fam"] = E.cpt.map(fam_of)
    H_cpt, H_fam = b.History(E.pat + "|" + E.cpt, E.day.values), b.History(E.pat + "|" + E.fam, E.day.values)
    H_any = b.History(E.pat, E.day.values)

    # history for test-window lines (2026 file; CSN needed for the join)
    t = pd.read_parquet(b.fs_path(b.V8D.format(year=2026))[1], columns=["CSN", "PAT_ID", "FINAL_CPT", "SERVICEDATE"],
                        filesystem=b.fs_path(b.V8D.format(year=2026))[0])
    t["FINAL_CPT"] = b.norm(t.FINAL_CPT)
    t = t[~t.FINAL_CPT.str.fullmatch(b.QUALITY)]
    day = b.days(t.SERVICEDATE)
    t = t[(day >= b.days(pd.Series(["2026-05-01"]))[0]) & (day < b.days(pd.Series(["2026-08-01"]))[0])]
    day = b.days(t.SERVICEDATE)
    pat = b.norm(t.PAT_ID)
    fam = t.FINAL_CPT.map(lambda c: fam_of.get(c) or b.family(c))
    t["cpt90"] = H_cpt.count(pat + "|" + t.FINAL_CPT, day, lo, 90)
    t["fam90"] = H_fam.count(pat + "|" + fam, day, lo, 90)
    t["any90"] = H_any.count(pat, day, lo, 90)
    t["fam365"] = H_fam.count(pat + "|" + fam, day, lo, 365)
    t["family"] = fam
    t["CSN"] = t.CSN.astype(str).str.replace(r"\.0$", "", regex=True)
    h = t.groupby(["CSN", "FINAL_CPT"]).agg(cpt90=("cpt90", "max"), fam90=("fam90", "max"), any90=("any90", "max"),
                                             fam365=("fam365", "max"), family=("family", "first"))

    # core model scores at service level
    s = pd.read_parquet(f"{a.run}/test_preds.parquet")
    s = s.loc[:, ~s.columns.duplicated()]
    pcol = [c for c in s.columns if c.startswith("p_")][0]
    s["CSN"] = s.CSN.astype(str).str.replace(r"\.0$", "", regex=True)
    s["FINAL_CPT"] = b.norm(s.FINAL_CPT)
    s = s[~s.FINAL_CPT.str.fullmatch(b.QUALITY)]
    S = s.groupby(["CSN", "FINAL_CPT"]).agg(y=("y", "max"), s=(pcol, "max")).join(h, how="left")
    for c in ["cpt90", "fam90", "any90", "fam365"]:
        S[c] = S[c].fillna(0)
    S["family"] = S.family.fillna("unknown")
    print(f"test services {len(S):,} | matched to history rows {S.index.isin(h.index).mean():.1%} | PA {int(S.y.sum()):,}")
    cut = float(np.quantile(S.s[S.y.eq(1)], 1 - a.recall))
    f = S.s.ge(cut)
    S["grp"] = np.select([f & S.y.eq(1), f & S.y.eq(0), ~f & S.y.eq(1)], ["TP", "FP", "FN"], "TN")
    S["has_hist"] = S.fam90.gt(0)
    pd.set_option("display.width", 220)

    print(f"\n== 1. PA rate with vs without family history (90 days), per family")
    r = S.groupby(["family", "has_hist"]).y.agg(services="size", pa_rate="mean").unstack("has_hist")
    r.columns = [f"{m}_{'hist' if h else 'no_hist'}" for m, h in r.columns]
    r = r.fillna(0).sort_values("services_no_hist", ascending=False)
    print(r.assign(**{c: r[c].map("{:.1%}".format) for c in r if c.startswith("pa_rate")}).head(15).to_string())

    print(f"\n== 2. Share with history, by outcome at {a.recall:.0%} recall (cut-off {cut:.4f})")
    sh = S.groupby("grp")[["cpt90", "fam90", "any90", "fam365"]].agg(lambda x: (x > 0).mean()).reindex(["TP", "FP", "FN", "TN"])
    sh.insert(0, "services", S.grp.value_counts().reindex(sh.index))
    print(sh.assign(**{c: sh[c].map("{:.1%}".format) for c in ["cpt90", "fam90", "any90", "fam365"]}).to_string())

    print("\n== 3. Per family: false alarms without history vs PA caught with history (family, 90 days)")
    G = pd.DataFrame({"family": S.family.values, "FP": S.grp.eq("FP").values, "TP": S.grp.eq("TP").values,
                      "FN": S.grp.eq("FN").values, "h": S.has_hist.values})
    G["FP_no_hist"], G["TP_with_hist"], G["FN_with_hist"] = G.FP & ~G.h, G.TP & G.h, G.FN & G.h
    g = G.groupby("family")[["FP", "FP_no_hist", "TP", "TP_with_hist", "FN_with_hist"]].sum()
    print(g.sort_values("FP", ascending=False).head(15).to_string())
    os._exit(0)


if __name__ == "__main__":
    main()
