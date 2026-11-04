"""
v6_label_impact.py - BEFORE building a v6 dataset: what would the two proposed label changes touch, and what could go wrong?

Proposed changes
  Q  quality codes (CPT Category II, ^\\d{4}F$)           -> FALSE
  P  "passenger" lines: TRUE only through an auth number that is also on another CPT of the same claim,
     for CPTs that are rarely PA on their own ("alone on claim" TRUE rate below a threshold)  -> FALSE

Nothing is written to the dataset. The script only measures:
  1. size        lines / TRUE lines / visits affected, per year and in the test window
  2. which codes the candidate passenger list at several thresholds (1%, 2%, 5%)
  3. risks
     R1 visits that would lose ALL their PA lines (visit-level label flips: an auth existed but no "PA" service is left)
     R2 payor exceptions: a listed CPT that IS often PA on its own at a specific payor (referral / HMO plans)
     R3 codes we cannot classify: too few "alone on claim" lines to estimate (add-on codes etc.)
     R4 stability: the alone rate by year (2025 labels are inflated by the AMBIGUOUS drop)
     R5 quality codes: how many lines (easy negatives) they add if kept as FALSE instead of removed

Claim = CSN + FILING_ORDER_C (the auth number is written per claim). "Alone" = the only CPT on the claim.
Reference window for the alone rates = lines before --test-start (training data only).

Run : nohup python -u v6_label_impact.py --years 2025 2026 --out v6_impact > run_v6_impact.log 2>&1 &
      (add 2022 2023 2024 to --years to size the change on the full training history; slower)
Out : v6_impact/*.csv  + summary printed to the log
"""
import argparse
import os

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.fs as pafs

import pa_base_adapter as pb

COLS = ["CSN", "SERVICEDATE", "FINAL_CPT", "PAYOR", "PLAN", "GROUP_LABEL", "LINE_AUTHORIZATION_NUMBER", "FILING_ORDER_C"]
BLANK = {"", "NAN", "NONE", "NULL", "<NA>"}
NO_AUTH = r"(NO\s*(REF|REFERRAL|AUTH|PA|PRIOR)\b.*|NOT\s*REQ.*|NAR|NPN|NPAR|N/?A|NONE|0+)"
QUALITY = r"^\d{4}F$"


def load_year(y, fs):
    d = ds.dataset(pb.BASE_PATH.format(year=y).replace("s3://", ""), filesystem=fs, format="parquet")
    cols = [c for c in COLS if c in d.schema.names]
    parts = []
    for b in d.to_batches(columns=cols, batch_size=500_000):
        t = b.to_pandas()
        t["FINAL_CPT"] = t.FINAL_CPT.astype(str).str.strip().str.upper()
        t["L"] = t.GROUP_LABEL.astype(str).str.strip().str.upper().map(
            {"TRUE": "TRUE", "1": "TRUE", "FALSE": "FALSE", "0": "FALSE"}).fillna("AMBIGUOUS")
        a = t.LINE_AUTHORIZATION_NUMBER.astype(object).fillna("").astype(str).str.strip().str.upper().str.rstrip(",. ")
        t["AUTH"] = a.where(~a.isin(BLANK) & ~a.str.fullmatch(NO_AUTH, na=False), "")
        t["SVC"] = pd.to_datetime(t.SERVICEDATE, errors="coerce")
        t["CLAIM"] = t.CSN.astype(str) + "|" + pd.to_numeric(t.get("FILING_ORDER_C", 1), errors="coerce").fillna(1).astype(int).astype(str)
        t = pb.clean_names(t)
        parts.append(t[["CSN", "CLAIM", "SVC", "FINAL_CPT", "PAYOR_CLEAN", "L", "AUTH"]])
    t = pd.concat(parts, ignore_index=True)
    for c in ["FINAL_CPT", "PAYOR_CLEAN", "L"]:
        t[c] = t[c].astype("category")
    t["YEAR"] = y
    return t


def claim_flags(t):
    """per line: n distinct CPTs on its claim, and whether its auth value is also on another CPT of the claim"""
    t["n_cpt_claim"] = t.groupby("CLAIM", observed=True).FINAL_CPT.transform("nunique")
    a = t[t.AUTH != ""][["CLAIM", "AUTH", "FINAL_CPT"]].drop_duplicates()
    n_cpt_auth = a.groupby(["CLAIM", "AUTH"], observed=True).FINAL_CPT.nunique().rename("n_cpt_auth")
    t = t.merge(n_cpt_auth, left_on=["CLAIM", "AUTH"], right_index=True, how="left")
    t["auth_shared"] = t.n_cpt_auth.fillna(0) >= 2
    t["src"] = np.select([(t.L == "TRUE") & (t.AUTH != ""), t.L == "TRUE"], ["TRUE via auth", "TRUE via code"], "not TRUE")
    return t.drop(columns="n_cpt_auth")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, default=[2025, 2026])
    ap.add_argument("--test-start", default="2026-05-01")
    ap.add_argument("--test-end", default="2026-08-01")
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.01, 0.02, 0.05])
    ap.add_argument("--min-alone", type=int, default=100, help="labelled 'alone on claim' lines needed to classify a CPT")
    ap.add_argument("--payor-min-alone", type=int, default=50)
    ap.add_argument("--payor-exception-rate", type=float, default=0.05)
    ap.add_argument("--rate-basis", choices=["labelled", "amb_false"], default="labelled",
                    help="alone rate on labelled lines only, or with AMBIGUOUS counted as not PA (removes 2025 inflation)")
    ap.add_argument("--ref-years", nargs="*", type=int, default=[], help="years used for the alone rate (default: all loaded)")
    ap.add_argument("--out", default="v6_impact")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 50)
    fs = pafs.S3FileSystem()
    ts, te = pd.Timestamp(a.test_start), pd.Timestamp(a.test_end)

    D = []
    for y in a.years:
        t = claim_flags(load_year(y, fs))
        D.append(t)
        print(f"[{y}] {len(t):,} lines | TRUE {int((t.L == 'TRUE').sum()):,}")
    D = pd.concat(D, ignore_index=True)
    D["is_quality"] = D.FINAL_CPT.astype(str).str.match(QUALITY)
    D["window"] = np.where(D.SVC >= te, "after test", np.where(D.SVC >= ts, "test", "train"))
    lab = D.L.isin(["TRUE", "FALSE"])

    # ------------------------------------------------------------------ Q: quality codes
    q = D[D.is_quality]
    qa = pd.DataFrame({
        "lines": D.groupby("YEAR").size(),
        "quality_lines": q.groupby("YEAR").size(),
        "quality_TRUE": q[q.L == "TRUE"].groupby("YEAR").size(),
        "all_TRUE": D[D.L == "TRUE"].groupby("YEAR").size(),
        "quality_labelled": q[q.L.isin(["TRUE", "FALSE"])].groupby("YEAR").size(),
        "all_labelled": D[lab].groupby("YEAR").size()}).fillna(0).astype(int)
    qa["quality_share_of_lines"] = (qa.quality_lines / qa.lines).round(3)
    qa["quality_share_of_TRUE"] = (qa.quality_TRUE / qa.all_TRUE).round(3)
    qa["quality_share_of_labelled"] = (qa.quality_labelled / qa.all_labelled).round(3)
    qa.to_csv(f"{a.out}/Q_quality_by_year.csv")
    print("\n== Q. quality codes by year\n", qa.to_string())
    print("\n== Q. how quality-code TRUE lines got TRUE\n",
          pd.crosstab(q.src, q.auth_shared.map({True: "auth shared", False: "auth not shared"}), margins=True).to_string())
    qt = q[q.L == "TRUE"]
    print("\n== Q. quality-code TRUE alone on claim (should be ~0):", int((qt.n_cpt_claim == 1).sum()))

    # ------------------------------------------------------------------ alone-on-claim rates (training window only)
    # rate basis: 'labelled' = TRUE / (TRUE+FALSE) (inflated where AMBIGUOUS lines were dropped, mostly 2025)
    #             'amb_false' = TRUE / all lines incl. AMBIGUOUS (AMBIGUOUS counted as not PA - no drop inflation)
    base = D[(D.window == "train") & ~D.is_quality]
    if a.ref_years:
        base = base[base.YEAR.isin(a.ref_years)]
    ref = (base if a.rate_basis == "amb_false" else base[base.L.isin(["TRUE", "FALSE"])]).copy()
    ref["y"] = ref.L.eq("TRUE")
    print(f"\n   alone-rate reference: basis={a.rate_basis}, years={a.ref_years or a.years}, {len(ref):,} lines")
    ref["ctx"] = np.where(ref.n_cpt_claim == 1, "alone", "with")
    r = ref.groupby(["FINAL_CPT", "ctx"], observed=True).y.agg(["sum", "size"]).unstack(fill_value=0)
    r.columns = [f"{s}_{c}" for s, c in r.columns]
    for c in ["sum_alone", "size_alone", "sum_with", "size_with"]:
        if c not in r:
            r[c] = 0
    r["rate_alone"] = r.sum_alone / r.size_alone.replace(0, np.nan)
    r["rate_with"] = r.sum_with / r.size_with.replace(0, np.nan)
    r["classifiable"] = r.size_alone >= a.min_alone

    # TRUE-via-shared-auth lines per CPT (what the passenger rule would flip), all loaded years
    flip_base = D[(D.src == "TRUE via auth") & D.auth_shared & ~D.is_quality]
    r["shared_auth_TRUE_lines"] = flip_base.groupby("FINAL_CPT", observed=True).size().reindex(r.index).fillna(0).astype(int)
    r.sort_values("shared_auth_TRUE_lines", ascending=False).to_csv(f"{a.out}/alone_rates_by_cpt.csv")

    # ------------------------------------------------------------------ P: passenger list at each threshold
    all_true = int((D.L == "TRUE").sum())
    rows, flips = [], {}
    for thr in a.thresholds:
        lst = set(r[r.classifiable & (r.rate_alone < thr)].index)
        f = flip_base[flip_base.FINAL_CPT.isin(lst)]
        flips[thr] = f
        test_f = f[f.window == "test"]
        rows.append({"threshold": thr, "codes_in_list": len(lst), "TRUE_lines_flipped": len(f),
                     "share_of_all_TRUE": round(len(f) / max(all_true, 1), 3),
                     "visits_touched": f.CSN.nunique(),
                     "test_TRUE_services_flipped": len(test_f[["CSN", "FINAL_CPT"]].drop_duplicates())})
        pd.Series(sorted(lst)).to_csv(f"{a.out}/P_list_thr{int(thr * 100)}.csv", index=False, header=["FINAL_CPT"])
    P = pd.DataFrame(rows)
    P.to_csv(f"{a.out}/P_size_by_threshold.csv", index=False)
    print(f"\n== P. passenger rule size (all TRUE lines in loaded years: {all_true:,})\n", P.to_string(index=False))
    mid = a.thresholds[len(a.thresholds) // 2]
    top = (flips[mid].groupby("FINAL_CPT", observed=True).size().sort_values(ascending=False).head(25)
           .rename("TRUE_lines_flipped").to_frame().join(r[["size_alone", "rate_alone", "size_with", "rate_with"]]))
    print(f"\n== P. top codes flipped at threshold {mid:.0%}\n", top.round(4).to_string())
    print(f"\n   flipped lines by year (threshold {mid:.0%}):\n", flips[mid].groupby("YEAR").size().to_string())
    watch = ["99211", "99212", "99213", "99214", "99215", "99203", "99204", "99205", "G2211", "36415", "93000", "96372"]
    print("\n   where the common office-visit / lab codes sit (alone rate vs thresholds):\n",
          r.loc[r.index.isin(watch), ["size_alone", "rate_alone", "size_with", "rate_with", "shared_auth_TRUE_lines"]]
          .sort_values("shared_auth_TRUE_lines", ascending=False).round(4).to_string())

    # ------------------------------------------------------------------ risks
    # R2: payor exceptions - listed CPT that is often PA alone at a specific payor (computed first, used in R1)
    lst_mid = set(r[r.classifiable & (r.rate_alone < mid)].index)
    rp = ref[ref.FINAL_CPT.isin(lst_mid) & (ref.ctx == "alone")].groupby(["FINAL_CPT", "PAYOR_CLEAN"], observed=True).y.agg(["mean", "size"])
    exc = rp[(rp["size"] >= a.payor_min_alone) & (rp["mean"] >= a.payor_exception_rate)].sort_values("size", ascending=False)
    exc = exc.join(flips[mid].groupby(["FINAL_CPT", "PAYOR_CLEAN"], observed=True).size().rename("TRUE_lines_flipped_here"))
    exc.to_csv(f"{a.out}/R2_payor_exceptions.csv")
    print(f"\n== R2. payor exceptions (listed CPT, alone-on-claim PA rate >= {a.payor_exception_rate:.0%} at that payor, "
          f">= {a.payor_min_alone} alone lines): {len(exc)}\n", exc.head(20).round(3).to_string())
    ex_pairs = set(exc.index)
    fm = flips[mid]
    keep_flip = ~pd.Series(list(zip(fm.FINAL_CPT.astype(str), fm.PAYOR_CLEAN.astype(str))), index=fm.index).isin(ex_pairs)
    fm_ex = fm[keep_flip.values]
    print(f"   flips at {mid:.0%} after excluding exception pairs: {len(fm_ex):,} TRUE lines "
          f"(was {len(fm):,}); test services {len(fm_ex[fm_ex.window == 'test'][['CSN', 'FINAL_CPT']].drop_duplicates()):,}")

    # R1: visits that lose ALL TRUE lines (Q + P at the middle threshold, exceptions excluded)
    t_true = D[D.L == "TRUE"]
    gone = pd.concat([t_true[t_true.is_quality], fm_ex]).index.unique()
    v = t_true.assign(flip=t_true.index.isin(gone)).groupby("CSN").flip.agg(["all", "size"])
    lost = v[v["all"]]
    print(f"\n== R1. visits whose ALL PA lines would be flipped (visit-level PA -> not PA): {len(lost):,} "
          f"of {len(v):,} PA visits ({len(lost) / max(len(v), 1):.1%})")
    lost_cpts = t_true[t_true.CSN.isin(lost.index)].groupby("FINAL_CPT", observed=True).size().sort_values(ascending=False).head(15)
    print("   CPTs on those visits:\n", lost_cpts.to_string())
    lost_cpts.to_csv(f"{a.out}/R1_visits_losing_all_PA_cpts.csv")
    lp = t_true[t_true.CSN.isin(lost.index)].groupby("PAYOR_CLEAN", observed=True).CSN.nunique().sort_values(ascending=False).head(10)
    print("   payors of those visits:\n", lp.to_string())

    # OWNER RULE: flip a passenger only if its claim keeps at least one TRUE line that is neither a quality code
    # nor itself flipped (= the auth has an 'owner'). Claims with no owner left = visit-level / referral auth -> keep TRUE.
    stays = t_true[~t_true.index.isin(gone)]
    owner_claims = set(stays.CLAIM)
    fo = fm_ex[fm_ex.CLAIM.isin(owner_claims)]
    no_owner = fm_ex[~fm_ex.CLAIM.isin(owner_claims)]
    print(f"\n== OWNER RULE (threshold {mid:.0%}, exceptions excluded): flip {len(fo):,} TRUE lines "
          f"({len(fo) / max(all_true, 1):.1%} of TRUE), test services {len(fo[fo.window == 'test'][['CSN', 'FINAL_CPT']].drop_duplicates()):,}; "
          f"kept TRUE as visit-level auth: {len(no_owner):,} lines on {no_owner.CLAIM.nunique():,} claims")
    print("   by year:\n", fo.groupby("YEAR").size().to_string())
    print("   top codes flipped:\n", fo.groupby("FINAL_CPT", observed=True).size().sort_values(ascending=False).head(12).to_string())
    print("   kept as visit-level auth, by payor:\n",
          no_owner.groupby("PAYOR_CLEAN", observed=True).size().sort_values(ascending=False).head(10).to_string())
    gone2 = pd.concat([t_true[t_true.is_quality], fo]).index.unique()
    v2 = t_true.assign(flip=t_true.index.isin(gone2)).groupby("CSN").flip.all()
    print(f"   R1 under owner rule: visits losing all PA = {int(v2.sum()):,} (only quality-code-only visits should remain)")
    fo[["CSN", "CLAIM", "FINAL_CPT", "PAYOR_CLEAN", "YEAR"]].to_csv(f"{a.out}/owner_rule_flips.csv", index=False)

    # R3: codes we cannot classify but that carry many shared-auth TRUE lines
    unc = r[~r.classifiable].sort_values("shared_auth_TRUE_lines", ascending=False)
    unc[["size_alone", "rate_alone", "size_with", "rate_with", "shared_auth_TRUE_lines"]].head(30).to_csv(f"{a.out}/R3_unclassifiable.csv")
    print(f"\n== R3. CPTs with < {a.min_alone} alone lines (left unchanged): {len(unc):,} codes, "
          f"{int(unc.shared_auth_TRUE_lines.sum()):,} shared-auth TRUE lines\n",
          unc[["size_alone", "rate_with", "shared_auth_TRUE_lines"]].head(15).round(3).to_string())

    # R4: stability of the alone rate by year for the listed codes
    st = ref[ref.FINAL_CPT.isin(lst_mid) & (ref.ctx == "alone")].groupby(["FINAL_CPT", "YEAR"], observed=True).y.mean().unstack()
    st = st.loc[st.index.isin(top.index)]
    st.to_csv(f"{a.out}/R4_alone_rate_by_year.csv")
    print("\n== R4. alone-on-claim PA rate by year (top flipped codes)\n", st.round(4).to_string())

    # R5: quality codes kept as FALSE vs removed - share of labelled training lines and of test services
    tq = D[(D.window == "test") & lab]
    svc = tq.groupby(["CSN", "FINAL_CPT"], observed=True).agg(q=("is_quality", "first"), y=("L", lambda s: (s == "TRUE").any()))
    print("\n== R5. test window services: quality vs other\n",
          svc.groupby("q").y.agg(services="size", PA_services="sum").to_string())
    print(f"\noutputs in {a.out}/")


if __name__ == "__main__":
    main()


#nohup python -u v6_label_impact.py --years 2025 2026 --rate-basis amb_false --thresholds 0.01 0.02 0.05 --out v6_impact_owner > run_v6_impact_owner.log 2>&1 &
