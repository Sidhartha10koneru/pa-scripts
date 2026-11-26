"""
rad_profile.py - how much radiology (CPT 70010-79999) is in the dataset, and what its labels look like.  Counts only.

Reads every year file, keeps radiology lines, and reports at SERVICE level (CSN + CPT; PA if any line is TRUE):
  P1_by_year         lines / services / visits, share of all lines, PA rate per label
  P2_by_window       same for TRAIN (< --update-start), UPDATE, TEST (>= --test-start) - the split our runs use
  P3_label_values    raw values of each label column on radiology lines (TRUE / FALSE / anything else)
  P4_agreement_*     service-level agreement between LABEL_V6_PAX and each other label (TRUE / FALSE / unknown)
  P5_subsection      AMA radiology subsections: diagnostic (CT / MRI / X-ray), ultrasound, guidance, mammography,
                     bone density, radiation oncology, nuclear medicine
  P6_filing_order    PA rate and volume by filing order of the line (primary vs secondary)
  P7_top_cpts        top 25 radiology CPTs by volume, PA rate per label
  P8_top_plans       top 20 payor | plan by volume, PA rate per label
  P9_monthly         PA rate by month per label (drift)
Service label: TRUE if any line TRUE; FALSE if no TRUE and at least one FALSE; otherwise "unknown" (left out of rates).

  nohup python -u rad_profile.py --out rad_profile > run_rad_profile.log 2>&1 &
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.fs as pafs

V8 = "s3://cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v8_draft/ai_prior_auth_09282026_v3_{year}_std.parquet"
LABELS = ["LABEL_V6_PAX", "LABEL_V7_STRICT", "LABEL_V7_STRICT_PAX", "LABEL_V6", "LABEL_V7", "LABEL_V7_PAX"]
BASE = ["CSN", "FINAL_CPT", "SERVICEDATE", "PAYOR", "PLAN", "FILING_ORDER_C", "PLACE_OF_SERVICE_CODE"]
SUBSECTIONS = [(70010, 76499, "1 diagnostic radiology (CT / MRI / X-ray)"), (76506, 76999, "2 ultrasound"),
               (77001, 77022, "3 radiologic guidance"), (77046, 77067, "4 mammography"), (77071, 77086, "5 bone / joint (incl. DXA)"),
               (77261, 77799, "6 radiation oncology"), (78012, 79999, "7 nuclear medicine")]
TRUE_V, FALSE_V = {"TRUE", "1", "1.0"}, {"FALSE", "0", "0.0"}


def key(s):
    return s.astype(object).where(s.notna(), None).map(lambda v: "" if v is None else str(v)) \
        .str.strip().str.replace(r"^(\d+)\.0$", r"\1", regex=True)


def clean_uniques(s):
    codes, uniq = pd.factorize(s.astype(object).fillna(""))
    u = pd.Series(uniq, dtype=object).astype(str).str.upper().str.replace(r"\bAS OF\b.*$", "", regex=True) \
        .str.replace(r"\s+", " ", regex=True).str.strip()
    return pd.Series(u.values[codes], index=s.index)


def subsection(n):
    for lo, hi, name in SUBSECTIONS:
        if lo <= n <= hi:
            return name
    return "8 other 70010-79999"


def rates(g, labs):
    """services, PA services and PA rate per label for a grouped service frame"""
    out = {"services": g.size()}
    for l in labs:
        known = g[l].apply(lambda s: (s != "unknown").sum())
        pa = g[l].apply(lambda s: (s == "TRUE").sum())
        out[f"{l}_PA"] = pa
        out[f"{l}_rate"] = (pa / known.replace(0, np.nan)).round(4)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-path", default=V8)
    ap.add_argument("--years", nargs="+", type=int, default=[2022, 2023, 2024, 2025, 2026])
    ap.add_argument("--update-start", default="2026-02-01")
    ap.add_argument("--test-start", default="2026-05-01")
    ap.add_argument("--cpt-from", type=int, default=70010)
    ap.add_argument("--cpt-to", type=int, default=79999)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 55)

    parts, totals, raw_vals = [], {}, []
    labs = None
    for y in a.years:
        t0 = time.time()
        p = a.base_path.format(year=y)
        fs, pp = (pafs.S3FileSystem(), p.replace("s3://", "")) if p.startswith("s3://") else (pafs.LocalFileSystem(), os.path.abspath(p))
        d = ds.dataset(pp, filesystem=fs, format="parquet")
        names = d.schema.names
        labs = labs or [l for l in LABELS if l in names]
        cols = [c for c in BASE if c in names] + labs
        totals[y] = d.count_rows()
        try:                                            # string range filter pushed into the read; exact check below
            x = d.to_table(columns=cols, filter=(ds.field("FINAL_CPT") >= str(a.cpt_from)) & (ds.field("FINAL_CPT") <= str(a.cpt_to))).to_pandas()
        except Exception:
            x = d.to_table(columns=cols).to_pandas()
        x["k_cpt"] = key(x.FINAL_CPT).str.upper()
        n = pd.to_numeric(x.k_cpt.where(x.k_cpt.str.fullmatch(r"\d{5}")), errors="coerce")
        x = x[n.between(a.cpt_from, a.cpt_to)].copy()
        x["cptn"] = n[x.index].astype(int)
        x["k_csn"] = key(x.CSN)
        x["dos"] = pd.to_datetime(x.SERVICEDATE, errors="coerce")
        x["plan"] = clean_uniques(x.PAYOR) + " | " + clean_uniques(x.PLAN)
        x["fo"] = key(x.FILING_ORDER_C) if "FILING_ORDER_C" in x else "?"
        for l in labs:
            v = key(x[l]).str.upper()
            raw_vals.append(v.value_counts().rename("lines").reset_index().rename(columns={l: "value", "index": "value"}).assign(label=l, year=y))
            x[l] = np.where(v.isin(TRUE_V), 2, np.where(v.isin(FALSE_V), 1, 0))      # 2 TRUE > 1 FALSE > 0 unknown
        keep = ["k_csn", "k_cpt", "cptn", "dos", "plan", "fo"] + labs
        parts.append(x[keep].assign(year=y))
        print(f"[{y}] all lines {totals[y]:,} | radiology lines {len(x):,} ({len(x) / max(totals[y], 1):.1%}) | {time.time() - t0:.0f}s")
        del x
    L = pd.concat(parts, ignore_index=True)

    # service level: one row per CSN + CPT (max over lines: TRUE beats FALSE beats unknown); primary line wins plan
    L["prim"] = L.fo.eq("1")
    L = L.sort_values(["prim"], ascending=False, kind="stable")
    S = L.groupby(["k_csn", "k_cpt"], sort=False).agg(**{l: (l, "max") for l in labs}, cptn=("cptn", "first"), dos=("dos", "min"),
                                                     plan=("plan", "first"), year=("year", "first"),
                                                     n_lines=("fo", "size"), has_sec=("prim", lambda s: bool((~s).any()))).reset_index()
    for l in labs:
        S[l] = S[l].map({2: "TRUE", 1: "FALSE", 0: "unknown"})
    S["window"] = np.select([S.dos < pd.Timestamp(a.update_start), S.dos < pd.Timestamp(a.test_start)], ["1 train", "2 update"], "3 test")
    S["sub"] = S.cptn.map(subsection)
    S["month"] = S.dos.dt.to_period("M").astype(str)
    print(f"\nradiology: {len(L):,} lines | {len(S):,} services | {S.k_csn.nunique():,} visits | labels found: {labs}")

    out = {}
    ly = L.groupby("year").size().rename("lines")
    p1 = rates(S.groupby("year"), labs)
    p1.insert(0, "lines", ly)
    p1.insert(1, "share_of_all_lines", (ly / pd.Series(totals)).round(4))
    p1.insert(3, "visits", S.groupby("year").k_csn.nunique())
    out["P1_by_year"] = p1
    p2 = rates(S.groupby("window"), labs)
    p2.insert(1, "visits", S.groupby("window").k_csn.nunique())
    out["P2_by_window"] = p2
    rv = pd.concat(raw_vals).groupby(["label", "value"]).lines.sum().unstack(0).fillna(0).astype(int)
    out["P3_label_values"] = rv.loc[rv.sum(axis=1).sort_values(ascending=False).index].head(15)
    for l in labs:
        if l != "LABEL_V6_PAX" and "LABEL_V6_PAX" in labs:
            out[f"P4_agreement_{l}"] = pd.crosstab(S.LABEL_V6_PAX.rename("LABEL_V6_PAX (rows)"), S[l], margins=True)
    p5 = rates(S.groupby("sub"), labs)
    p5.insert(1, "share", (p5.services / len(S)).round(4))
    out["P5_subsection"] = p5
    fo = L.groupby("fo").size().rename("lines")
    sec = S.groupby("has_sec").size().rename("services")
    out["P6_filing_order"] = pd.concat([fo, (fo / fo.sum()).round(4).rename("share")], axis=1)
    lab0 = labs[0]
    Lr = L.assign(lab=L[lab0].map({2: "TRUE", 1: "FALSE", 0: "unknown"}))
    fo_rate = Lr[Lr.lab != "unknown"].groupby("fo").lab.apply(lambda s: round((s == "TRUE").mean(), 4)).rename(f"{lab0}_line_rate")
    out["P6_filing_order"] = out["P6_filing_order"].join(fo_rate)
    out["P6b_services_with_secondary"] = rates(S.groupby("has_sec"), labs).assign(share=lambda z: (z.services / len(S)).round(4))
    top = S.k_cpt.value_counts().head(25).index
    out["P7_top_cpts"] = rates(S[S.k_cpt.isin(top)].groupby("k_cpt"), labs).sort_values("services", ascending=False)
    topp = S.plan.value_counts().head(20).index
    out["P8_top_plans"] = rates(S[S.plan.isin(topp)].groupby("plan"), labs).sort_values("services", ascending=False)
    p9 = rates(S.groupby("month"), labs)
    out["P9_monthly"] = p9[["services"] + [c for c in p9 if c.endswith("_rate")]]

    for k, v in out.items():
        v.to_csv(f"{a.out}/{k}.csv")
        print(f"\n== {k}\n{v.to_string()}")
    print(f"\noutputs -> {a.out}/ (counts and rates only)")
    os._exit(0)


if __name__ == "__main__":
    main()
