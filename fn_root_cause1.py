"""
fn_root_cause.py - WHY is each FN service an FN?  Concrete, rule-based evidence per FN (no model).

Every FN service (CSN + CPT, PA in test, scored below the 80%-recall cut-off) is run through the same checks.
Each check is a plain lookup in the data and writes its evidence (values, rates, counts) next to the FN:

  T1 label source      [V4 2026] what made this service TRUE: auth number / PA denial code / another remit row
  T2 passenger         [V4 2026] the auth number is also on another CPT in the same visit (CSN), and that CPT
                       is much more often PA (owner).  -> LABEL: TRUE belongs to another service
  T3 open auth         [V4 2025-26] the same auth number was already on an EARLIER visit of this patient
                       -> DATA: an authorisation existed before this visit (auth / episode history not a feature)
  T4 subgroup rule     [V4, before test start] inside the FN's own cell, the FN's value of a pre-visit column
                       (specialty, department, provider, PCP, visit type, primary dx, visit number, therapy visit
                       number) had a high PA rate in training (>= the cut-off, so the model would have flagged it, and >= 1.5x the cell rate, n >= MIN_N)
                       -> DATA: a concrete rule we can state ("SPECIALTY = X: 68% PA, n = 40")
  T5 rate change       [V4] the cell's PA rate in the test window is >= SHIFT_X x its training rate
                       -> CHANGE: new rule / plan change after training
  T6 too little data   the cell had < MIN_PA PA lines before the test window
  T7 inflation         [V4 incl. AMBIGUOUS] cell training rate as labelled vs if AMBIGUOUS = FALSE (reported only;
                       inflation raises scores, so it explains FP more than FN)

Primary cause = first that applies:  T2 passenger > T3 open auth > T4 subgroup > T5 change > T6 too little > unexplained
All flags are kept as columns, so overlaps can be counted.

Inputs : exp_nofilter_2022/fn_buckets_recall80_min20.csv   (cell 8 of the bucket notebook: one row per FN service)
Outputs: <out>/fn_causes.csv     one row per FN with all evidence
         <out>/cause_rollup.csv  FN by bucket x primary cause
         <out>/cells.csv         per cell: FN, cause mix, best subgroup evidence, rates
Run    : nohup python -u fn_root_cause.py --buckets 4 --out rc_mixed > run_rc_mixed.log 2>&1 &
         python fn_root_cause.py --buckets 1 2 3 4 5 --cover 0.8 --out rc_all
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.fs as pafs

import pa_base_adapter as pb

KEY = ["FINAL_CPT", "FINAL_MODIFIERS", "PLACE_OF_SERVICE_CODE", "PAYOR_PLAN_CLEAN"]
THERAPY = re.compile(r"^(97\d{3}|9894[0-2]|92507|92526)$")
THERAPY_CODES = ["97110", "97112", "97116", "97140", "97530", "97535", "97161", "97162", "97163", "97164", "97165",
                 "97166", "97167", "97168", "98940", "98941", "98942", "92507", "92526"]
ATTRS = ["SPECIALTY", "DEPARTMENT", "DEPARTMENT_ID", "SERVICEPROVIDERNPI", "PCPNPI", "VISITTYPE", "Primary_DX",
         "cell_visit_bin", "therapy_visit_bin"]
BLANK = {"", "NAN", "NONE", "NULL", "<NA>"}
NO_AUTH = r"(NO\s*(REF|REFERRAL|AUTH|PA)\b.*|NOT\s*REQ.*|NAR|N/?A|NONE|0+)"
VBINS = [0, 1, 2, 5, 10, 25, 10_000]
VLABS = ["1", "2", "3-5", "6-10", "11-25", "26+"]


# ----------------------------------------------------------------------------------------------- loading helpers
def dset(year, fs):
    return ds.dataset(pb.BASE_PATH.format(year=year).replace("s3://", ""), filesystem=fs, format="parquet")


def isin(d, col, vals):
    """pyarrow isin filter with the values cast to the column's type"""
    t = d.schema.field(col).type
    vals = [str(v).strip() for v in vals if str(v).strip().upper() not in BLANK]
    if pa.types.is_integer(t):
        vals = [int(v) if v.isdigit() else int(float(v)) for v in vals]
    elif pa.types.is_floating(t):
        vals = [float(v) for v in vals]
    return ds.field(col).isin(sorted(set(vals)))


NEED = ["CSN", "PAT_ID", "SERVICEDATE", "FILING_ORDER_C", "PAYOR", "PLAN", "FINAL_CPT", "FINAL_MODIFIERS",
        "PLACE_OF_SERVICE_CODE", "GROUP_LABEL", "LEVEL1_LABEL", "LEVEL2_LABEL", "LINE_AUTHORIZATION_NUMBER"] + ATTRS


def read(d, filt, row_filter=None):
    """read in batches, keep only the columns we use, filter rows per batch -> low memory"""
    out = []
    for b in d.to_batches(filter=filt, batch_size=250_000):
        t = b.to_pandas()
        t = t.loc[:, ~t.columns.duplicated()]
        t = prep(t[[c for c in NEED if c in t]])
        if row_filter is not None:
            t = row_filter(t)
        if len(t):
            out.append(t)
    if not out:
        return pd.DataFrame()
    t = pd.concat(out, ignore_index=True)
    for c in ["SPECIALTY", "DEPARTMENT", "DEPARTMENT_ID", "SERVICEPROVIDERNPI", "PCPNPI", "VISITTYPE", "Primary_DX",
              "PAYOR", "PLAN", "PAYOR_PLAN_CLEAN", "FINAL_MODIFIERS", "PLACE_OF_SERVICE_CODE", "L", "GROUP_LABEL",
              "LEVEL1_LABEL", "LEVEL2_LABEL"]:
        if c in t:
            t[c] = t[c].astype(str).astype("category")
    return t


def prep(t):
    t = pb.clean_names(t.loc[:, ~t.columns.duplicated()])
    for c in KEY:
        t[c] = t[c].astype(object).fillna("").astype(str).str.strip()
    t["L"] = t.GROUP_LABEL.astype(str).str.strip().str.upper().map(
        {"TRUE": "TRUE", "1": "TRUE", "FALSE": "FALSE", "0": "FALSE"}).fillna("AMBIGUOUS")
    t["SVC"] = pd.to_datetime(t.SERVICEDATE, errors="coerce")
    t["CSN"] = t.CSN.astype(str).str.replace(r"\.0$", "", regex=True)
    if "PAT_ID" in t:
        t["PAT_ID"] = t.PAT_ID.astype(str).str.replace(r"\.0$", "", regex=True)
    a = t.get("LINE_AUTHORIZATION_NUMBER", pd.Series("", index=t.index)).astype(object).fillna("").astype(str)
    a = a.str.strip().str.upper()
    t["AUTH"] = a.where(~a.isin(BLANK) & ~a.str.fullmatch(NO_AUTH, na=False), "")
    t["primary"] = pd.to_numeric(t.get("FILING_ORDER_C", 1), errors="coerce").fillna(1).eq(1)
    return t[t.SVC.notna()]


def cpt_rates(years, test_start, fs):
    """[V4] PA rate per CPT before the test window (labelled lines) - used to find the owner of a shared auth"""
    out = []
    for y in years:
        d = dset(y, fs)
        if y < test_start.year:                      # whole year is before the test window: aggregate in arrow
            t = d.to_table(columns=["FINAL_CPT", "GROUP_LABEL"])
            lab = pc.utf8_upper(pc.cast(t["GROUP_LABEL"], pa.string()))
            t = t.append_column("yy", pc.cast(pc.is_in(lab, pa.array(["TRUE", "1"])), pa.int64()))
            t = t.filter(pc.is_in(lab, pa.array(["TRUE", "1", "FALSE", "0"])))
            g = t.group_by("FINAL_CPT").aggregate([("yy", "sum"), ("yy", "count")]).to_pandas()
        else:                                        # test year: date filter in pandas
            gs = []
            for b in d.to_batches(columns=["FINAL_CPT", "GROUP_LABEL", "SERVICEDATE"], batch_size=500_000):
                t = b.to_pandas()
                lab = t.GROUP_LABEL.astype(str).str.strip().str.upper()
                t = t[lab.isin(["TRUE", "1", "FALSE", "0"]) & (pd.to_datetime(t.SERVICEDATE, errors="coerce") < test_start)]
                t = t.assign(yy=lab[t.index].isin(["TRUE", "1"]).astype(int))
                gs.append(t.groupby("FINAL_CPT").yy.agg(yy_sum="sum", yy_count="size").reset_index())
            g = pd.concat(gs)
        out.append(g)
    g = pd.concat(out)
    g["FINAL_CPT"] = g.FINAL_CPT.astype(str).str.strip()
    g = g.groupby("FINAL_CPT")[["yy_sum", "yy_count"]].sum()
    g.index = g.index.astype(str).str.strip()
    return (g.yy_sum / g.yy_count).rename("cpt_rate"), g.yy_count.rename("cpt_n")


# ----------------------------------------------------------------------------------------------- checks
def visit_bins(cell_rows, therapy_rows):
    """visit number this calendar year per patient: in the cell, and over the plan's therapy codes"""
    v = cell_rows[["PAT_ID", "SVC"]].drop_duplicates().sort_values(["PAT_ID", "SVC"])
    v["n"] = v.groupby(["PAT_ID", v.SVC.dt.year]).cumcount() + 1
    cell_rows = cell_rows.merge(v, on=["PAT_ID", "SVC"], how="left")
    cell_rows["cell_visit_bin"] = pd.cut(cell_rows.n, VBINS, labels=VLABS).astype(str)
    cell_rows = cell_rows.drop(columns="n")
    if therapy_rows is not None and len(therapy_rows):
        w = therapy_rows[["PAT_ID", "SVC"]].drop_duplicates().sort_values(["PAT_ID", "SVC"])
        w["n"] = w.groupby(["PAT_ID", w.SVC.dt.year]).cumcount() + 1
        cell_rows = cell_rows.merge(w, on=["PAT_ID", "SVC"], how="left")
        cell_rows["therapy_visit_bin"] = pd.cut(cell_rows.n, VBINS, labels=VLABS).astype(str)
        cell_rows = cell_rows.drop(columns="n")
    return cell_rows


def subgroup(fit, fnr, cut, min_n, rel_x):
    """T4: for each FN, the pre-visit column value with the highest training PA rate that clears the bar"""
    cell_rate = fit.y.mean()
    bar = max(cut, rel_x * cell_rate)
    best = pd.DataFrame(index=fnr.index, data={"t4_attr": "", "t4_value": "", "t4_rate": np.nan, "t4_n": np.nan})
    for a in ATTRS:
        if a not in fit or a not in fnr:
            continue
        r = fit.assign(v=fit[a].astype(str)).groupby("v").y.agg(rate="mean", n="size")
        r = r[(r.n >= min_n) & (r.rate >= bar) & (~r.index.isin(["nan", "None", "NA", ""]))]
        if r.empty:
            continue
        hit = fnr[a].astype(str).map(r.rate)
        better = hit.notna() & (hit > best.t4_rate.fillna(-1))
        best.loc[better, "t4_attr"] = a
        best.loc[better, "t4_value"] = fnr.loc[better, a].astype(str)
        best.loc[better, "t4_rate"] = hit[better]
        best.loc[better, "t4_n"] = fnr.loc[better, a].astype(str).map(r.n)
    return best, cell_rate


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fn-file", default="exp_nofilter_2022/fn_buckets_recall80_min20.csv")
    ap.add_argument("--buckets", nargs="+", default=["4"], help="bucket number prefixes, e.g. 4  or  1 2 3 4 5")
    ap.add_argument("--cover", type=float, default=1.0, help="biggest cells until this share of the FN")
    ap.add_argument("--max-cells", type=int, default=300)
    ap.add_argument("--cut", type=float, default=0.421, help="operating cut-off (80%% recall) of the model")
    ap.add_argument("--years", nargs="+", type=int, default=[2025, 2026])
    ap.add_argument("--test-start", default="2026-05-01")
    ap.add_argument("--test-end", default="2026-08-01")
    ap.add_argument("--min-n", type=int, default=10, help="T4: lines needed for a subgroup value")
    ap.add_argument("--rel-x", type=float, default=1.5, help="T4: subgroup rate must also be >= this x cell rate")
    ap.add_argument("--owner-x", type=float, default=2.0, help="T2: owner CPT rate must be >= this x FN CPT rate")
    ap.add_argument("--shift-x", type=float, default=2.0, help="T5: test rate >= this x training rate")
    ap.add_argument("--min-pa", type=int, default=10, help="T6: fewer PA lines than this before test = too little")
    ap.add_argument("--out", default="rc_mixed")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 90)
    fs = pafs.S3FileSystem()
    ts, te = pd.Timestamp(a.test_start), pd.Timestamp(a.test_end)

    # ---- FN services in scope
    fn = pd.read_csv(a.fn_file, dtype={**{k: str for k in KEY}, "CSN": str})
    for c in KEY:
        fn[c] = fn[c].fillna("").astype(str).str.strip()
    fn["CSN"] = fn.CSN.astype(str).str.replace(r"\.0$", "", regex=True)
    fn = fn[fn.bucket.astype(str).str[0].isin([b[0] for b in a.buckets])].copy()
    cs = fn.groupby(KEY).size().sort_values(ascending=False).rename("fn_services").reset_index()
    sh = cs.fn_services.cumsum() / cs.fn_services.sum()
    cs = cs[sh.shift(fill_value=0) < a.cover].head(a.max_cells)
    fn = fn.merge(cs[KEY], on=KEY)
    print(f"FN in scope: {len(fn):,} services in {len(cs)} cells (buckets {a.buckets})")

    # ---- [V4] cells: all lines of the FN cells (+ therapy codes of the same plans)
    cpts, plans = set(cs.FINAL_CPT), set(cs.PAYOR_PLAN_CLEAN)
    if any(THERAPY.match(c) for c in cpts):
        cpts |= set(THERAPY_CODES)
    pairs = set(zip(cs.FINAL_CPT, cs.PAYOR_PLAN_CLEAN))
    ther_plans = {p for c, p in pairs if THERAPY.match(c)}

    def keep_cells(t):                       # only the (CPT, plan) pairs of the FN cells (+ therapy codes of therapy plans)
        k = pd.Series(list(zip(t.FINAL_CPT, t.PAYOR_PLAN_CLEAN)), index=t.index).isin(pairs)
        k |= t.FINAL_CPT.isin(THERAPY_CODES) & t.PAYOR_PLAN_CLEAN.isin(ther_plans)
        return t[k & (t.SVC < te)]

    parts = []
    for y in a.years:
        parts.append(read(dset(y, fs), ds.field("FINAL_CPT").isin(sorted(cpts)), keep_cells))
        print(f"  [V4] cells {y}: {len(parts[-1]):,} lines")
    cell_all = pd.concat(parts, ignore_index=True)

    # ---- [V4] patients: every line (any CPT) of the FN patients - for T2 (same visit) and T3 (earlier visits)
    fn_csn = set(fn.CSN)
    pats = set(cell_all.loc[cell_all.CSN.isin(fn_csn), "PAT_ID"]) if "PAT_ID" in cell_all else set()
    parts = []
    for y in a.years:
        d = dset(y, fs)
        if pats and "PAT_ID" in d.schema.names:
            parts.append(read(d, isin(d, "PAT_ID", pats)))
    pat = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    print(f"  [V4] patients: {len(pats):,} FN patients, {len(pat):,} lines")

    cpt_rate, cpt_n = cpt_rates(a.years, ts, fs)

    # ---- T1 / T2 / T3 per FN service (from the patient frame)
    if len(pat):
        tl = pat[pat.CSN.isin(fn_csn) & (pat.L == "TRUE")]
        l1 = tl.get("LEVEL1_LABEL", pd.Series("", index=tl.index)).astype(str).str.upper().eq("TRUE")
        l2 = tl.get("LEVEL2_LABEL", pd.Series("", index=tl.index)).astype(str).str.upper().eq("TRUE")
        tl = tl.assign(src=np.select([l1 & tl.AUTH.ne(""), l2], ["auth number", "PA denial code"], "other remit row"))
        src = tl.groupby(["CSN", "FINAL_CPT"]).agg(
            t1_source=("src", lambda s: min(set(s), key=["auth number", "PA denial code", "other remit row"].index)),
            auths=("AUTH", lambda s: sorted(set(s) - {""})))
        fn = fn.merge(src, left_on=["CSN", "FINAL_CPT"], right_index=True, how="left")
        fn["auths"] = fn.auths.apply(lambda v: v if isinstance(v, list) else [])

        # T2: same auth on another CPT in the same visit
        csn_auth = pat[pat.CSN.isin(fn_csn) & pat.AUTH.ne("")][["CSN", "FINAL_CPT", "AUTH"]].drop_duplicates()
        sharers = csn_auth.groupby(["CSN", "AUTH"]).FINAL_CPT.apply(set).to_dict()
        own_r = fn.FINAL_CPT.map(cpt_rate).fillna(0)
        t2_with, t2_owner, t2_owner_rate = [], [], []
        for csn, cpt, auths, r0 in zip(fn.CSN, fn.FINAL_CPT, fn.auths, own_r):
            others = set().union(*[sharers.get((csn, x), set()) for x in auths]) - {cpt} if auths else set()
            t2_with.append(",".join(sorted(others))[:80])
            if others:
                o = max(others, key=lambda c: cpt_rate.get(c, 0))
                t2_owner.append(o); t2_owner_rate.append(cpt_rate.get(o, np.nan))
            else:
                t2_owner.append(""); t2_owner_rate.append(np.nan)
        fn["t2_shared_with"], fn["t2_owner"], fn["t2_owner_rate"] = t2_with, t2_owner, t2_owner_rate
        fn["fn_cpt_rate"] = own_r.values
        fn["T2_passenger"] = (fn.t2_owner.ne("") & (fn.t2_owner_rate >= a.owner_x * fn.fn_cpt_rate)
                              & (fn.t2_owner_rate >= 0.10))

        # T3: auth already seen on an earlier date for this patient
        first_seen = pat[pat.AUTH.ne("")].groupby(["PAT_ID", "AUTH"]).SVC.min().to_dict()
        svc = pat[pat.CSN.isin(fn_csn)].groupby("CSN").agg(PAT_ID=("PAT_ID", "first"), SVC=("SVC", "min"))
        fn = fn.merge(svc, left_on="CSN", right_index=True, how="left")
        days = []
        for p, d0, auths in zip(fn.PAT_ID, fn.SVC, fn.auths):
            ds_ = [(d0 - first_seen[(p, x)]).days for x in auths if (p, x) in first_seen]
            days.append(max(ds_) if ds_ else np.nan)
        fn["t3_auth_age_days"] = days
        fn["T3_open_auth"] = fn.t3_auth_age_days > 0
    else:
        print("!! no PAT_ID / patient lines - T1-T3 skipped")
        for c in ["T2_passenger", "T3_open_auth"]:
            fn[c] = False

    # ---- T4 / T5 / T6 / T7 per cell
    fn["T4_subgroup"] = False
    cell_rows = []
    for _, c in cs.iterrows():
        m = np.logical_and.reduce([cell_all[k] == c[k] for k in KEY])
        call = cell_all[m]
        prim = call[call.primary]
        lab = prim[prim.L.isin(["TRUE", "FALSE"])].assign(y=lambda t: (t.L == "TRUE").astype(int))
        ther = None
        if THERAPY.match(c.FINAL_CPT):
            ther = cell_all[(cell_all.PAYOR_PLAN_CLEAN == c.PAYOR_PLAN_CLEAN) & cell_all.FINAL_CPT.str.match(THERAPY.pattern)
                            & cell_all.primary]
        vb = None
        if "PAT_ID" in call and len(prim):
            vb = visit_bins(prim, ther)
            vb = vb[["CSN", "FINAL_CPT", "SVC"] + [x for x in ["cell_visit_bin", "therapy_visit_bin"] if x in vb]]
            vb = vb.drop_duplicates(["CSN", "FINAL_CPT", "SVC"])
            lab = lab.merge(vb, on=["CSN", "FINAL_CPT", "SVC"], how="left")
        fit, tst = lab[lab.SVC < ts], lab[lab.SVC >= ts]
        r = {**{k: c[k] for k in KEY}, "fn_services": int(c.fn_services), "fit_lines": len(fit),
             "fit_pa": int(fit.y.sum()), "fit_rate": fit.y.mean() if len(fit) else np.nan,
             "test_rate": tst.y.mean() if len(tst) else np.nan}
        pf = prim[prim.SVC < ts]
        r["fit_rate_if_amb_false"] = (pf.L == "TRUE").mean() if len(pf) else np.nan
        r["T7_inflation_x"] = r["fit_rate"] / r["fit_rate_if_amb_false"] if r["fit_rate_if_amb_false"] else np.nan
        mon = lab.groupby(lab.SVC.dt.to_period("M")).y.agg(["mean", "size"])
        r["monthly_rate"] = " ".join(f"{p.strftime('%y%m')}:{v:.0%}/{n}" for p, (v, n) in mon.tail(10).iterrows())
        r["first_pa_month"] = str(lab[lab.y == 1].SVC.min())[:7] if lab.y.any() else ""
        idx = fn.index[np.logical_and.reduce([fn[k] == c[k] for k in KEY])]
        # FN rows with their own pre-visit values (their TRUE line in this cell)
        fl = call[call.CSN.isin(set(fn.loc[idx, "CSN"])) & (call.L == "TRUE")]
        if vb is not None:
            fl = fl.merge(vb, on=["CSN", "FINAL_CPT", "SVC"], how="left")
        fl = fl.drop_duplicates("CSN").set_index("CSN")
        fnr = fn.loc[idx, ["CSN"]].join(fl[[x for x in ATTRS if x in fl]], on="CSN")
        if len(fit) and fit.y.sum() and len(fnr):
            best, _ = subgroup(fit, fnr, a.cut, a.min_n, a.rel_x)
            for col in best:
                fn.loc[idx, col] = best[col].values
            fn.loc[idx, "T4_subgroup"] = best.t4_attr.ne("").values
            top = best[best.t4_attr.ne("")].groupby(["t4_attr", "t4_value"]).agg(fn=("t4_rate", "size"),
                                                                                 rate=("t4_rate", "first"),
                                                                                 n=("t4_n", "first"))
            r["t4_rules"] = "; ".join(f"{i[0]}={i[1][:30]}: {v.rate:.0%} n={int(v.n)} ({v.fn} FN)"
                                      for i, v in top.sort_values("fn", ascending=False).head(4).iterrows())
        r["T5_change"] = bool(np.isfinite(r["fit_rate"]) and np.isfinite(r["test_rate"])
                              and r["test_rate"] >= a.shift_x * max(r["fit_rate"], 1e-9))
        r["T6_too_little"] = r["fit_pa"] < a.min_pa
        fn.loc[idx, "T5_change"] = r["T5_change"]
        fn.loc[idx, "T6_too_little"] = r["T6_too_little"]
        fn.loc[idx, "cell_fit_rate"], fn.loc[idx, "cell_test_rate"] = r["fit_rate"], r["test_rate"]
        cell_rows.append(r)
        print(f"{c.FINAL_CPT} {c.FINAL_MODIFIERS or '-'} {c.PLACE_OF_SERVICE_CODE} {c.PAYOR_PLAN_CLEAN[:40]:<40} "
              f"FN {int(c.fn_services):>4} | fit {r['fit_rate']:.1%} test {r['test_rate']:.1%} | {r.get('t4_rules', '')[:90]}")

    for c in ["T2_passenger", "T3_open_auth", "T4_subgroup", "T5_change", "T6_too_little"]:
        fn[c] = fn[c].fillna(False).astype(bool)
    fn["cause"] = np.select(
        [fn.T2_passenger, fn.T3_open_auth, fn.T4_subgroup, fn.T5_change, fn.T6_too_little],
        ["LABEL: passenger of another service's auth",
         "DATA: auth already open from an earlier visit",
         "DATA: subgroup rule (pre-visit column)",
         "CHANGE: cell PA rate rose after training",
         "INSUFFICIENT: too few PA in training"],
        "UNEXPLAINED by these checks")

    cells = pd.DataFrame(cell_rows)
    mix = pd.crosstab([fn[k] for k in KEY], fn.cause)
    cells = cells.merge(mix, left_on=KEY, right_index=True, how="left")
    fn.drop(columns=["auths"], errors="ignore").to_csv(f"{a.out}/fn_causes.csv", index=False)
    cells.to_csv(f"{a.out}/cells.csv", index=False)
    roll = pd.crosstab(fn.cause, fn.bucket, margins=True)
    roll["share"] = (roll["All"] / len(fn)).round(3)
    roll.to_csv(f"{a.out}/cause_rollup.csv")

    print("\n== primary cause (FN services)\n", roll.to_string())
    print("\n== label source of the FN (T1)\n", fn.get("t1_source", pd.Series(dtype=str)).value_counts(dropna=False).to_string())
    print("\n== overlaps: how many FN pass each check (not exclusive)\n",
          fn[["T2_passenger", "T3_open_auth", "T4_subgroup", "T5_change", "T6_too_little"]].sum().to_string())
    if fn.T2_passenger.any():
        print("\n== passenger owners (FN CPT -> owner CPT)\n",
              fn[fn.T2_passenger].groupby(["FINAL_CPT", "t2_owner"]).size().sort_values(ascending=False).head(15).to_string())
    print(f"\noutputs in {a.out}/: fn_causes.csv, cells.csv, cause_rollup.csv")


if __name__ == "__main__":
    main()
