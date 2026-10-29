"""
pa_diagnostics.py - why does the PA model behave the way it does? model limit vs feature limit vs label limit.

Reads the test predictions saved by pa_experiments.py baselines (test_preds.parquet) and the model's
category lists. No retraining.

Sections
  1. CEILING   - best possible score using ONLY today's features (cell = CPT+mod+POS+payor|plan):
                 leave-one-out cell rate on the test set itself. If the model is close to this, a better
                 algorithm will not help much -> need new features or cleaner labels.
                 Also: irreducible error = lines that disagree with their own cell's majority.
  2. ENTROPY   - how much label uncertainty is left once the cell is known; which big cells are mixed.
  3. SEGMENTS  - where the model is good/bad: CPT family, payor segment, POS missing, rules value,
                 unknown CPT/plan (not in the model's category list), cell size.
  4. ERRORS    - at the 90%-recall cut-off: what the missed PA visits (FN) and false alarms (FP) look like.
  5. RIDE-ALONG- inside PA visits, how many lines score low (labels copied to every line of the visit).
  6. CALIBRATION by score decile.

Run:
  python pa_diagnostics.py --preds exp_d1_full/test_preds.parquet --cats exp_d1_full/models/lgb_rules_full_categories.json \
      --model p_lgb_rules_full --out diag_full 2>&1 | tee run_diag_full.log
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, log_loss

CSN, CPT, MOD, POS, PAYOR, PLAN = "CSN", "FINAL_CPT", "FINAL_MODIFIERS", "PLACE_OF_SERVICE_CODE", "PAYOR_CLEAN", "PAYOR_PLAN_CLEAN"
CELL = [CPT, MOD, POS, PLAN]


def cpt_family(c):
    c = str(c).strip().upper()
    if not c or c in ("NAN", "NONE", "<NA>"):
        return "missing"
    if c[0].isalpha():
        return {"J": "HCPCS J drugs", "A": "HCPCS A supplies/contrast", "Q": "HCPCS Q", "G": "HCPCS G",
                "E": "HCPCS E/K/L DME", "K": "HCPCS E/K/L DME", "L": "HCPCS E/K/L DME"}.get(c[0], "HCPCS other")
    if c.endswith("F"):
        return "Cat II (quality)"
    if c.endswith("T"):
        return "Cat III (emerging)"
    if c.endswith("U"):
        return "PLA lab"
    try:
        n = int(c[:5])
    except ValueError:
        return "other"
    if n < 2000:
        return "anesthesia"
    if n < 70000:
        return "surgery"
    if n < 80000:
        return "imaging"
    if n < 90000:
        return "lab/path"
    if 97010 <= n <= 97799:
        return "PT/OT rehab"
    if 96360 <= n <= 96549:
        return "infusion/chemo admin"
    if 90785 <= n <= 90899:
        return "psych"
    if 92920 <= n <= 93799:
        return "cardio (echo/cath/ECG)"
    if 99202 <= n <= 99499:
        return "E/M visits"
    return "medicine other"


def payor_segment(p):
    p = str(p).upper()
    for seg, words in [("Medicare (traditional)", ["MEDICARE PART", "^MEDICARE$"]),
                       ("Medicare Advantage", ["MEDICARE ADVANTAGE", "AARP", "MEDICARE REPLACEMENT", "MEDIBLUE", "SENIOR"]),
                       ("Medicaid", ["MEDICAID", "HEALTHPLUS", "COMMUNITY PLAN", "CHIP", "FIDELIS", "HEALTHFIRST", "METROPLUS"]),
                       ("Supplement", ["SUPPLEMENT", "MEDIGAP"]),
                       ("Self-pay / other", ["SELF", "COMP", "WORKERS", "NO FAULT"]),
                       ("Tricare / VA", ["TRICARE", "CHAMPVA", "VA "])]:
        if any((w.startswith("^") and p == w.strip("^$")) or (not w.startswith("^") and w in p) for w in words):
            return seg
    return "Commercial / other"


def H(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def visit(df, col):
    return df.groupby(CSN).agg(y=("y", "max"), p=(col, "max"))


def ap(y, p):
    return average_precision_score(y, p) if 0 < np.mean(y) < 1 else np.nan


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--preds", default="exp_d1_full/test_preds.parquet")
    a.add_argument("--cats", default="exp_d1_full/models/lgb_rules_full_categories.json")
    a.add_argument("--model", default="p_lgb_rules_full")
    a.add_argument("--recall", type=float, default=0.90)
    a.add_argument("--out", default="diag_full")
    o = a.parse_args()
    os.makedirs(o.out, exist_ok=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60); pd.set_option("display.max_rows", 200)

    t = pd.read_parquet(o.preds)
    for c in CELL + [PAYOR]:
        t[c] = t[c].astype(str)
    t["p"] = t[o.model]
    t["cell"] = t[CELL].agg("|".join, axis=1)
    t["family"] = t[CPT].map(cpt_family)
    t["segment"] = t[PLAN].map(payor_segment)
    print(f"Test: {len(t):,} lines, {t[CSN].nunique():,} visits, PA line rate {t.y.mean():.3%}, model {o.model}")

    # ------------------------------------------------------------------ 1. CEILING
    g = t.groupby("cell").y.agg(["sum", "count"])
    t = t.join(g, on="cell")
    gm = t.y.mean()
    # leave-one-out cell rate, smoothed towards the overall rate (m=5) so 1-line cells are not 0/0
    t["p_oracle"] = (t["sum"] - t.y + 5 * gm) / (t["count"] - 1 + 5)
    vm, vo = visit(t, "p"), visit(t, "p_oracle")
    maj_err = (g[["sum", "count"]].apply(lambda r: min(r["sum"], r["count"] - r["sum"]), axis=1).sum()) / len(t)
    ceil = pd.DataFrame({
        "AP_visit": [ap(vm.y, vm.p), ap(vo.y, vo.p)], "AUC_visit": [roc_auc_score(vm.y, vm.p), roc_auc_score(vo.y, vo.p)],
        "AP_line": [ap(t.y, t.p), ap(t.y, t.p_oracle)]}, index=["model", "ceiling (cell rate, leave-one-out, on test)"])
    print("\n== 1. CEILING: best achievable with today's features (cell = CPT+mod+POS+payor|plan)\n", ceil.round(4).to_string())
    print(f"   Irreducible line error with these features: {maj_err:.2%} of lines disagree with their own cell's majority "
          f"(PA rate is {gm:.2%}, so this is {maj_err / gm:.0%} of the PA count)")
    print("   Reading: model close to ceiling -> new algorithm won't help much; need new features / cleaner labels.")
    ceil.to_csv(f"{o.out}/1_ceiling.csv")

    # ------------------------------------------------------------------ 2. ENTROPY
    gc = g.assign(rate=g["sum"] / g["count"])
    gc["H"] = H(gc.rate)
    H0, Hc = H(gm), (gc["count"] * gc.H).sum() / gc["count"].sum()
    print(f"\n== 2. ENTROPY: label uncertainty {H0:.4f} bits overall, {Hc:.4f} after knowing the cell "
          f"-> cells explain {1 - Hc / H0:.0%} of it")
    gc["pa_lines"] = gc["sum"]
    gc["uncertainty_share"] = gc["count"] * gc.H / (gc["count"] * gc.H).sum()
    mixed = gc[gc["count"] >= 200].sort_values("uncertainty_share", ascending=False).head(25)
    print("   Cells holding the most remaining uncertainty (>=200 lines):\n", mixed[["count", "pa_lines", "rate", "H", "uncertainty_share"]].round(4).to_string())
    gc.sort_values("uncertainty_share", ascending=False).to_csv(f"{o.out}/2_cell_entropy.csv")
    print(f"   Top 100 cells hold {gc.sort_values('uncertainty_share', ascending=False).uncertainty_share.head(100).sum():.0%} "
          f"of remaining uncertainty")

    # ------------------------------------------------------------------ 3. SEGMENTS
    cats = json.load(open(o.cats)) if os.path.exists(o.cats) else {}
    t["cpt_known"] = t[CPT].isin(set(cats.get(CPT, []))) if cats else np.nan
    t["plan_known"] = t[PLAN].isin(set(cats.get(PLAN, []))) if cats else np.nan
    t["pos_missing"] = t[POS].isin(["nan", "None", "<NA>", ""])
    t["rules"] = pd.cut(t.get("RULE_PLAN_SHARE", pd.Series(np.nan, index=t.index)).fillna(-1),
                        [-2, -0.5, 0.001, 0.999, 1.01], labels=["no rules match", "not listed", "partly listed", "listed by all"])
    t["cell_size"] = pd.cut(t["count"], [0, 20, 100, 1000, 1e9], labels=["<20", "20-100", "100-1000", ">1000"])

    def seg_table(col):
        rows = []
        for k, d in t.groupby(col, observed=True):
            v = visit(d, "p")
            rows.append({col: k, "lines": len(d), "line_share": len(d) / len(t), "PA_lines": int(d.y.sum()),
                         "PA_rate": d.y.mean(), "share_of_all_PA": d.y.sum() / t.y.sum(),
                         "mean_pred": d.p.mean(), "AP_line": ap(d.y, d.p), "AP_visit": ap(v.y, v.p)})
        return pd.DataFrame(rows).sort_values("PA_lines", ascending=False)

    for col in ["family", "segment", "rules", "pos_missing", "cpt_known", "plan_known", "cell_size"]:
        if t[col].isna().all():
            continue
        s = seg_table(col)
        print(f"\n== 3. SEGMENT: {col}\n", s.round(4).to_string(index=False))
        s.to_csv(f"{o.out}/3_segment_{col}.csv", index=False)

    # ------------------------------------------------------------------ 4. ERRORS at the recall cut-off
    v = vm.sort_values("p", ascending=False)
    caught = v.y.cumsum() / v.y.sum()
    k = int(np.argmax(caught.values >= o.recall))
    cut = v.p.iloc[k]
    v["flag"] = v.p >= cut
    print(f"\n== 4. ERRORS at {o.recall:.0%} recall: cut-off {cut:.4f}, flagged {v.flag.mean():.1%} of visits "
          f"({v.flag.sum():,}), FP {int((v.flag & (v.y == 0)).sum()):,}, FN {int((~v.flag & (v.y == 1)).sum()):,}")
    fn_csn = set(v.index[~v.flag & (v.y == 1)])
    tp_csn = set(v.index[v.flag & (v.y == 1)])
    fp_csn = set(v.index[v.flag & (v.y == 0)])
    top_line = t.sort_values("p", ascending=False).drop_duplicates(CSN).set_index(CSN)   # each visit's highest-scoring line

    def describe(csns, name):
        d = top_line.loc[list(csns)]
        fam = d.family.value_counts(normalize=True).head(10)
        seg = d.segment.value_counts(normalize=True).head(6)
        cp = t[t[CSN].isin(csns)].groupby(CPT).size().sort_values(ascending=False).head(15)
        print(f"\n   {name}: {len(csns):,} visits | top-line CPT family:\n{fam.round(3).to_string()}\n"
              f"   payor segment:\n{seg.round(3).to_string()}\n   most common CPTs on these visits:\n{cp.to_string()}")
        return d

    describe(fn_csn, "MISSED PA (FN)").to_csv(f"{o.out}/4_fn_visits_topline.csv")
    describe(fp_csn, "FALSE ALARMS (FP)").to_csv(f"{o.out}/4_fp_visits_topline.csv")
    describe(tp_csn, "CAUGHT PA (TP) for comparison")
    lines_per = t.groupby(CSN).size()
    print("\n   lines per visit - FN:", round(lines_per.loc[list(fn_csn)].mean(), 2), "| TP:", round(lines_per.loc[list(tp_csn)].mean(), 2),
          "| FP:", round(lines_per.loc[list(fp_csn)].mean(), 2))
    print("   FN visits whose cell had <20 test lines:", f"{(top_line.loc[list(fn_csn), 'count'] < 20).mean():.1%}",
          "| CPT unknown to model:", f"{(~top_line.loc[list(fn_csn), 'cpt_known'].astype(bool)).mean():.1%}" if cats else "")

    # ------------------------------------------------------------------ 5. RIDE-ALONG
    pa = t[t.y == 1]
    vmax = pa.groupby(CSN).p.transform("max")
    pa = pa.assign(rel=pa.p / vmax.clip(lower=1e-9))
    multi = pa[pa.groupby(CSN)[CSN].transform("size") > 1]
    print(f"\n== 5. RIDE-ALONG: PA visits with >1 line: {multi[CSN].nunique():,}. Share of their lines scoring <10% of the "
          f"visit's top line: {(multi.rel < 0.1).mean():.1%}")
    ride = multi[multi.rel < 0.1].groupby("family").size().sort_values(ascending=False).head(10)
    print("   low-scoring lines inside PA visits, by CPT family (likely labelled TRUE only because the visit had an auth):\n",
          ride.to_string())

    # ------------------------------------------------------------------ 6. CALIBRATION
    vm["decile"] = pd.qcut(vm.p.rank(method="first"), 10, labels=False) + 1
    cal = vm.groupby("decile").agg(visits=("y", "size"), mean_pred=("p", "mean"), actual=("y", "mean"))
    print("\n== 6. CALIBRATION by visit score decile (10 = highest)\n", cal.round(4).to_string())
    print(f"   visit log-loss {log_loss(vm.y, vm.p.clip(1e-6, 1 - 1e-6)):.4f} vs predicting the base rate "
          f"{log_loss(vm.y, np.full(len(vm), vm.y.mean())):.4f}")
    cal.to_csv(f"{o.out}/6_calibration.csv")
    print(f"\nSaved to {o.out}/")


if __name__ == "__main__":
    main()


#python pa_diagnostics.py --preds exp_d1_full/test_preds.parquet --cats exp_d1_full/models/lgb_rules_full_categories.json --model p_lgb_rules_full --out diag_full 2>&1 | tee run_diag_full.log
