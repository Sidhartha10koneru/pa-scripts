"""
full_review.py - full error review of the best model vs the baseline, for the team presentation.

Same test services as all runs (CSN + CPT, quality codes out); 'agreed' services = v8 passengers left out.
Cut-off per model = the score that catches --recall (85%) of PA services. Sections (all written to --out):

  1  overview          TP / FP / FN / TN, precision, at 80 / 85 / 90% recall, service AND visit level, both models
  2  by_family         CPT family (E/M, imaging, therapy, infusion, drugs, ...): PA rate, TP / FP / FN, both models
                       -> where the new fields removed false alarms, and where errors remain
  3  flags             label-noise signs per group (TP / FP / FN / TN): another PA service in the visit, auth text,
                       Acuity auth-denial code, PA under V7 strict, visit-level auth
  4  fp_top_* fn_top_* top CPT / visit type / payer / specialty for false alarms and for missed PA
  5  fn_distance       how far below the cut-off the missed PA services are (near misses vs deep misses)
  6  missed_visits     visit level: PA visits missed, by number of PA services in the visit
  7  samples           fp_review_sample.csv and fn_review_sample.csv for SME review (no raw auth values, no patient fields)

  python full_review.py --run exp_feat_core_s42 --base exp_v8d_l6pax_s42 --out full_review_core
Outputs stay on EFS (claims-level rows): share only with people who have claims-data access.
"""
import argparse
import os

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.fs as pafs

V9D = ("cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v9_draft/"
       "ai_prior_auth_09282026_v3_{year}_std.parquet")
QUALITY = r"^\d{4}F$"
TRUE_SET = {"TRUE", "1", "1.0"}
EMPTY = {"", "NAN", "NONE", "NULL", "<NA>"}
COLS = ["CSN", "FINAL_CPT", "LINE_AUTHORIZATION_NUMBER", "ACUITY_AUTH_CODES", "LABEL_V7_STRICT", "IS_PASSENGER",
        "VISIT_LEVEL_AUTH", "VISITTYPE", "DEPARTMENT_ID", "SPECIALTY"]
FLAGS = ["visit_has_other_pa", "auth_text", "acuity_auth_code", "v7_strict_pa", "visit_level_auth"]
OUT = {}                                                  # sheet name -> table (also saved as csv)


def csn(s):
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()


def filled(s):
    return ~s.astype(str).str.strip().str.upper().isin(EMPTY)


def family(c):
    c = str(c).strip().upper()
    if not c:
        return "unknown"
    if c[0].isalpha():
        return {"J": "Drugs (J codes)", "Q": "HCPCS Q (contrast, misc.)", "G": "HCPCS G (procedures)",
                "A": "HCPCS A (supplies, transport)", "E": "HCPCS E (equipment)", "L": "HCPCS L (orthotics)"}.get(
            c[0], "HCPCS other")
    if c.endswith("T"):
        return "Category III (T codes)"
    if not c[:5].isdigit():
        return "other"
    n = int(c[:5])
    for lo, hi, name in [(99202, 99499, "E/M visits"), (100, 1999, "Anesthesia"), (10004, 69990, "Surgery"),
                         (70010, 79999, "Imaging / radiology"), (80047, 89398, "Lab"),
                         (96360, 96549, "Infusion / injection admin"), (97010, 97799, "Therapy (PT/OT)"),
                         (92920, 93799, "Cardiovascular (echo, ECG)"), (90281, 99199, "Other medicine")]:
        if lo <= n <= hi:
            return name
    return "other"


def services(run):
    t = pd.read_parquet(f"{run}/test_preds.parquet")
    t = t.loc[:, ~t.columns.duplicated()]
    p = [c for c in t.columns if c.startswith("p_")][0]
    t["CSN"], t["FINAL_CPT"] = csn(t.CSN), t.FINAL_CPT.astype(str).str.strip().str.upper()
    t = t[~t.FINAL_CPT.str.fullmatch(QUALITY)]
    keep = [c for c in ["PAYOR_CLEAN", "PAYOR_PLAN_CLEAN", "PLACE_OF_SERVICE_CODE", "Primary_DX"] if c in t]
    return t.groupby(["CSN", "FINAL_CPT"]).agg(y=("y", "max"), s=(p, "max"), **{c: (c, "first") for c in keep})


def load_flags(want, years, pattern):
    parts = []
    for y in years:
        path = pattern.format(year=y)
        fs = None if os.path.exists(path) else pafs.S3FileSystem()
        d = ds.dataset(path, filesystem=fs, format="parquet")
        cols = [c for c in COLS if c in d.schema.names]
        for b in d.to_batches(columns=cols, batch_size=1_000_000):
            x = b.to_pandas()
            x["CSN"] = csn(x.CSN)
            x = x[x.CSN.isin(want)]
            if len(x):
                parts.append(x)
    f = pd.concat(parts, ignore_index=True)
    f["FINAL_CPT"] = f.FINAL_CPT.astype(str).str.strip().str.upper()
    f["auth_text"] = filled(f.LINE_AUTHORIZATION_NUMBER).astype(int) if "LINE_AUTHORIZATION_NUMBER" in f else 0
    f["acuity_auth_code"] = filled(f.ACUITY_AUTH_CODES).astype(int) if "ACUITY_AUTH_CODES" in f else 0
    f["v7_strict_pa"] = (f.LABEL_V7_STRICT.astype(str).str.strip().str.upper().isin(TRUE_SET).astype(int)
                         if "LABEL_V7_STRICT" in f else 0)
    for c in ["IS_PASSENGER", "VISIT_LEVEL_AUTH"]:
        f[c] = pd.to_numeric(f.get(c, 0), errors="coerce").fillna(0).astype(int)
    return f.groupby(["CSN", "FINAL_CPT"]).agg(
        auth_text=("auth_text", "max"), acuity_auth_code=("acuity_auth_code", "max"), v7_strict_pa=("v7_strict_pa", "max"),
        passenger=("IS_PASSENGER", "max"), visit_level_auth=("VISIT_LEVEL_AUTH", "max"),
        **{c: (c, "first") for c in ["VISITTYPE", "DEPARTMENT_ID", "SPECIALTY"] if c in f})


def confusion(y, s, recall):
    cut = float(np.quantile(s[y.eq(1)], 1 - recall))
    f = s.ge(cut)
    tp, fp = int((f & y.eq(1)).sum()), int((f & y.eq(0)).sum())
    fn, tn = int((~f & y.eq(1)).sum()), int((~f & y.eq(0)).sum())
    return cut, f, dict(TP=tp, FP=fp, FN=fn, TN=tn, precision=round(tp / max(tp + fp, 1), 4),
                        recall=round(tp / max(tp + fn, 1), 4), flagged=tp + fp)


def groups(y, f):
    return pd.Series(np.select([f & y.eq(1), f & y.eq(0), ~f & y.eq(1)], ["TP", "FP", "FN"], "TN"), index=y.index)


def top(S, grp, col, which, n=25):
    R = S.reset_index().assign(hit=grp.values == which, tp=grp.values == "TP")
    t = R.groupby(col).agg(**{which: ("hit", "sum")}, pa_caught=("tp", "sum"), services=("y", "size"),
                           pa_rate=("y", "mean")).sort_values(which, ascending=False)
    t[f"share_of_all_{which}"] = t[which] / max(int((grp == which).sum()), 1)
    t["cumulative_share"] = t[f"share_of_all_{which}"].cumsum()
    return t.head(n)


def pct(t, cols):
    return t.assign(**{c: t[c].map(lambda v: f"{v:.1%}") for c in cols if c in t})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="exp_feat_core_s42", help="the model to review")
    ap.add_argument("--base", default="exp_v8d_l6pax_s42", help="baseline model (5 fields, same labels)")
    ap.add_argument("--recall", type=float, default=0.85)
    ap.add_argument("--years", nargs="+", type=int, default=[2026])
    ap.add_argument("--data", default=V9D)
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--out", default="full_review_core")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pd.set_option("display.width", 250)
    lines = []                                              # summary.txt

    def say(x=""):
        print(x)
        lines.append(str(x))

    # ---------------------------------------------------------------- data
    M, B = services(a.run), services(a.base)
    idx = M.index.intersection(B.index)
    S = M.loc[idx].copy()
    S["s_base"] = B.s.loc[idx]
    if not (S.y == B.y.loc[idx]).all():
        say(f"WARNING: test labels differ between {a.run} and {a.base} on {(S.y != B.y.loc[idx]).sum():,} services")
    S = S.join(load_flags(set(idx.get_level_values(0)), a.years, a.data), how="left")
    for c in ["auth_text", "acuity_auth_code", "v7_strict_pa", "passenger", "visit_level_auth"]:
        S[c] = S[c].fillna(0).astype(int)
    S["family"] = [family(c) for c in S.index.get_level_values(1)]
    n_all = len(S)
    S = S[S.passenger.eq(0)]                                # agreed services
    S["visit_has_other_pa"] = (S.groupby(level=0).y.transform("sum") - S.y).gt(0).astype(int)
    say(f"run {a.run} vs baseline {a.base}")
    say(f"test services {n_all:,} | agreed (v8 passengers out) {len(S):,} | PA services {int(S.y.sum()):,} | "
        f"PA visits {int(S.groupby(level=0).y.max().sum()):,}")

    # ---------------------------------------------------------------- 1 overview
    rows = []
    for r in (0.80, 0.85, 0.90):
        for name, col in (("baseline", "s_base"), ("model", "s")):
            cut, f, c = confusion(S.y, S[col], r)
            v = pd.DataFrame({"y": S.y.values, "f": f.values}, index=S.index.get_level_values(0)).groupby(level=0).max()
            vtp, vfp = int((v.f & v.y.eq(1)).sum()), int((v.f & v.y.eq(0)).sum())
            vfn = int((~v.f & v.y.eq(1)).sum())
            rows.append({"recall target": f"{r:.0%}", "model": name, "cut-off": round(cut, 4), **c,
                         "visit_TP": vtp, "visit_FP": vfp, "visit_FN": vfn,
                         "visit_precision": round(vtp / max(vtp + vfp, 1), 4), "visit_recall": round(vtp / max(vtp + vfn, 1), 4)})
    OUT["1_overview"] = ov = pd.DataFrame(rows)
    say("\n== 1. Overview (service = CSN + CPT; visit = CSN, flagged if any service is above the cut-off)")
    say(ov.to_string(index=False))

    cut, f = confusion(S.y, S.s, a.recall)[:2]
    cut_b, f_b = confusion(S.y, S.s_base, a.recall)[:2]
    G, Gb = groups(S.y, f), groups(S.y, f_b)
    n = G.value_counts()
    say(f"\nReview at {a.recall:.0%} recall: model cut-off {cut:.4f} | TP {n.get('TP', 0):,} FP {n.get('FP', 0):,} "
        f"FN {n.get('FN', 0):,} TN {n.get('TN', 0):,}")

    # ---------------------------------------------------------------- 2 by CPT family
    R = pd.DataFrame({"family": S.family.values, "y": S.y.values, "g": G.values, "gb": Gb.values})
    fam = R.groupby("family").agg(services=("y", "size"), pa=("y", "sum"), pa_rate=("y", "mean"),
                                  FP_base=("gb", lambda g: (g == "FP").sum()), FP_model=("g", lambda g: (g == "FP").sum()),
                                  FN_base=("gb", lambda g: (g == "FN").sum()), FN_model=("g", lambda g: (g == "FN").sum()))
    fam["FP_change"] = fam.FP_model - fam.FP_base
    fam["share_of_model_FP"] = fam.FP_model / max(fam.FP_model.sum(), 1)
    fam["share_of_model_FN"] = fam.FN_model / max(fam.FN_model.sum(), 1)
    fam = fam.sort_values("FP_model", ascending=False)
    OUT["2_by_family"] = fam
    say(f"\n== 2. By CPT family at {a.recall:.0%} recall (baseline vs model) - where false alarms were removed / remain")
    say(pct(fam, ["pa_rate", "share_of_model_FP", "share_of_model_FN"]).to_string())

    # ---------------------------------------------------------------- 3 flags
    fl = S.groupby(G.values)[FLAGS].mean().reindex(["TP", "FP", "FN", "TN"])
    fl.insert(0, "services", G.value_counts().reindex(fl.index))
    OUT["3_flags"] = fl
    say("\n== 3. Label-noise signs per group (a sign common in FP but rare in TN -> those FPs may be real PA)")
    say(pct(fl, FLAGS).to_string())
    fp, fn = S[G.eq("FP")], S[G.eq("FN")]
    sign = fp[["auth_text", "acuity_auth_code", "v7_strict_pa"]].max(axis=1).eq(1)
    say(f"False alarms with an auth sign on the line: {int(sign.sum()):,} ({sign.mean():.1%}) | in a visit with another PA "
        f"service: {int(fp.visit_has_other_pa.sum()):,} ({fp.visit_has_other_pa.mean():.1%}) | neither: "
        f"{int((~sign & fp.visit_has_other_pa.eq(0)).sum()):,} ({(~sign & fp.visit_has_other_pa.eq(0)).mean():.1%})")
    say(f"Missed PA with an Acuity auth-denial code: {int(fn.acuity_auth_code.sum()):,} ({fn.acuity_auth_code.mean():.1%}) "
        f"vs PA caught {S[G.eq('TP')].acuity_auth_code.mean():.1%}  (denied / missing-auth PA lines)")

    # ---------------------------------------------------------------- 4 top tables
    for which, tag in (("FP", "fp"), ("FN", "fn")):
        for col in ["FINAL_CPT", "VISITTYPE", "PAYOR_CLEAN", "SPECIALTY"]:
            if col != "FINAL_CPT" and col not in S:
                continue
            t = top(S, G, col, which)
            OUT[f"4_{tag}_top_{col}"] = t
            say(f"\n== 4. Top 10 {col} by {which} (top 10 = {t[f'share_of_all_{which}'].head(10).sum():.1%} of all {which})")
            say(pct(t.head(10), ["pa_rate", f"share_of_all_{which}", "cumulative_share"]).to_string())

    # ---------------------------------------------------------------- 5 how far below the cut-off
    ratio = fn.s / cut
    bins = pd.cut(ratio, [0, 0.25, 0.5, 0.75, 0.9, 1.0], labels=["< 25% of cut-off", "25-50%", "50-75%", "75-90%",
                                                                   "90-100% (near miss)"], include_lowest=True)
    dist = bins.value_counts().sort_index().to_frame("missed_PA")
    dist.index.name = "score vs cut-off"
    dist["share"] = dist.missed_PA / max(len(fn), 1)
    OUT["5_fn_distance"] = dist
    say(f"\n== 5. Missed PA: score as a share of the cut-off ({cut:.4f})")
    say(pct(dist, ["share"]).to_string())

    # ---------------------------------------------------------------- 6 missed visits
    V = pd.DataFrame({"y": S.y.values, "f": f.values}, index=S.index.get_level_values(0)).groupby(level=0).agg(
        pa_services=("y", "sum"), caught=("f", "max"), any_pa=("y", "max"))
    V = V[V.any_pa.eq(1)]
    V["pa_services_in_visit"] = V.pa_services.clip(upper=5).map(lambda k: f"{k}" if k < 5 else "5+")
    mv = V.groupby("pa_services_in_visit").agg(pa_visits=("caught", "size"), missed=("caught", lambda c: (~c.astype(bool)).sum()))
    mv["miss_rate"] = mv.missed / mv.pa_visits
    OUT["6_missed_visits"] = mv
    say(f"\n== 6. Visit level: PA visits {len(V):,}, missed {int((~V.caught.astype(bool)).sum()):,} - by number of PA services in the visit")
    say(pct(mv, ["miss_rate"]).to_string())

    # ---------------------------------------------------------------- 7 samples for SME review
    drop = ["y", "passenger", "s_base"]
    for which, d in (("fp", fp), ("fn", fn)):
        d = d.sort_values("s", ascending=(which == "fn"))      # FP: highest scores first; FN: lowest scores first
        k = a.sample // 2
        smp = pd.concat([d.head(k), d.iloc[k:].sample(min(a.sample - k, max(len(d) - k, 0)), random_state=1)])
        smp = smp.reset_index().drop(columns=[c for c in drop if c in smp.columns]).rename(columns={"s": "score"})
        smp.insert(0, "why_in_sample", ["extreme score"] * min(k, len(smp)) + ["random"] * (len(smp) - min(k, len(smp))))
        smp["SME_really_PA (Y/N/unsure)"], smp["SME_comment"] = "", ""
        smp.to_csv(f"{a.out}/{which}_review_sample.csv", index=False)

    # ---------------------------------------------------------------- save
    for k, t in OUT.items():
        t.to_csv(f"{a.out}/{k}.csv")
    try:
        with pd.ExcelWriter(f"{a.out}/full_review.xlsx") as w:
            for k, t in OUT.items():
                t.to_excel(w, sheet_name=k[:31])
        say(f"\nSaved {a.out}/full_review.xlsx (one sheet per table), the same tables as csv, summary.txt, "
            f"fp_review_sample.csv, fn_review_sample.csv")
    except Exception as e:                                   # openpyxl missing: csv files are still there
        say(f"\n(xlsx not written: {e}) Saved csv tables, summary.txt and the review samples to {a.out}/")
    open(f"{a.out}/summary.txt", "w").write("\n".join(lines) + "\n")
    os._exit(0)


if __name__ == "__main__":
    main()
