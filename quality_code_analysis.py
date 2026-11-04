"""
quality_code_analysis.py - complete picture of quality codes (CPT Category II, ^\\d{4}F$) before changing anything.

Same preparation as the model pipeline (pa_base_adapter.prep): all filing orders kept, AMBIGUOUS dropped,
repeated lines (CSN + CPT + modifier + date + filing order) removed keeping the TRUE copy.

Three modes, compared side by side
  keep     current labels (benchmark)
  false    quality codes labelled FALSE whatever the original label (AMBIGUOUS quality lines are kept as FALSE)
  exclude  quality codes removed

Outputs (per year, plus the test window May-Jul 2026)
  1_distinct_codes.csv        every quality code: lines, labelled, TRUE, by year
  2_raw_counts.csv            raw file: lines / quality lines / labels of quality lines
  3_modes_by_year.csv         lines, PA lines, PA line rate; services (CSN+CPT), PA service rate; visits, PA visit rate
  4_test_window_modes.csv     the same for the test window
  5_visits_flipped.csv        visits whose ONLY PA lines are quality codes (visit-level PA lost in 'false'/'exclude')

Run : nohup python -u quality_code_analysis.py --years 2022 2023 2024 2025 2026 --out qc_analysis > run_qc_analysis.log 2>&1 &
"""
import argparse
import os

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.fs as pafs

import pa_base_adapter as pb

COLS = ["CSN", "SERVICEDATE", "FINAL_CPT", "FINAL_MODIFIERS", "FILING_ORDER_C", "GROUP_LABEL"]
QUALITY = r"^\d{4}F$"


def load_year(y, fs):
    d = ds.dataset(pb.BASE_PATH.format(year=y).replace("s3://", ""), filesystem=fs, format="parquet")
    cols = [c for c in COLS if c in d.schema.names]
    parts = []
    for b in d.to_batches(columns=cols, batch_size=1_000_000):
        t = b.to_pandas()
        t["FINAL_CPT"] = t.FINAL_CPT.astype(str).str.strip().str.upper()
        t["MOD"] = (t["FINAL_MODIFIERS"] if "FINAL_MODIFIERS" in t else pd.Series("", index=t.index)).astype(object).fillna("").astype(str).str.strip().str.upper()
        t["L"] = t.GROUP_LABEL.astype(str).str.strip().str.upper().map(
            {"TRUE": "TRUE", "1": "TRUE", "1.0": "TRUE", "FALSE": "FALSE", "0": "FALSE", "0.0": "FALSE"}).fillna("AMBIGUOUS")
        t["FO"] = pd.to_numeric(t["FILING_ORDER_C"] if "FILING_ORDER_C" in t else pd.Series(1, index=t.index), errors="coerce").fillna(-1).astype(int)
        t["SVC"] = pd.to_datetime(t.SERVICEDATE, errors="coerce")
        t["CSN"] = t.CSN.astype(str)
        parts.append(t[["CSN", "SVC", "FINAL_CPT", "MOD", "FO", "L"]])
    t = pd.concat(parts, ignore_index=True)
    t["q"] = t.FINAL_CPT.str.fullmatch(QUALITY)
    for c in ["FINAL_CPT", "MOD", "L"]:
        t[c] = t[c].astype("category")
    return t


def apply_mode(t, mode):
    """pipeline order: label -> quality rule -> AMBIGUOUS drop -> dedup keeping TRUE"""
    y = t.L.map({"TRUE": 1, "FALSE": 0, "AMBIGUOUS": np.nan}).astype(float)
    x = t.assign(y=y)
    if mode == "false":
        x.loc[x.q, "y"] = 0
    elif mode == "exclude":
        x = x[~x.q]
    x = x[x.y.notna()]
    x = x.sort_values("y", ascending=False, kind="stable").drop_duplicates(["CSN", "FINAL_CPT", "MOD", "SVC", "FO"])
    return x


def summarise(x, tag):
    s = x.groupby(["CSN", "FINAL_CPT"], observed=True).y.max()
    v = x.groupby("CSN").y.max()
    return {"scope": tag, "lines": len(x), "PA_lines": int(x.y.sum()), "PA_line_rate": x.y.mean(),
            "services": len(s), "PA_services": int(s.sum()), "PA_service_rate": s.mean(),
            "visits": len(v), "PA_visits": int(v.sum()), "PA_visit_rate": v.mean(),
            "quality_lines_in_data": int(x.q.sum()), "quality_PA_lines": int(x[x.q].y.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, default=[2022, 2023, 2024, 2025, 2026])
    ap.add_argument("--test-start", default="2026-05-01")
    ap.add_argument("--test-end", default="2026-08-01")
    ap.add_argument("--out", default="qc_analysis")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    fs = pafs.S3FileSystem()
    ts, te = pd.Timestamp(a.test_start), pd.Timestamp(a.test_end)

    codes, raw, modes, test_rows, flips = [], [], [], [], []
    for y in a.years:
        t = load_year(y, fs)
        q = t[t.q]
        # 1 distinct codes
        c = q.groupby("FINAL_CPT", observed=True).L.value_counts().unstack(fill_value=0)
        c = c.reindex(columns=["TRUE", "FALSE", "AMBIGUOUS"], fill_value=0).assign(lines=lambda z: z.sum(1), YEAR=y)
        codes.append(c.reset_index())
        # 2 raw counts
        raw.append({"YEAR": y, "lines": len(t), "quality_lines": len(q), "quality_share": len(q) / len(t),
                    "distinct_quality_codes": q.FINAL_CPT.nunique(),
                    **{f"quality_{k}": int((q.L == k).sum()) for k in ["TRUE", "FALSE", "AMBIGUOUS"]},
                    "all_TRUE": int((t.L == "TRUE").sum())})
        # 3 / 4 modes
        for m in ["keep", "false", "exclude"]:
            x = apply_mode(t, m)
            modes.append({"YEAR": y, "mode": m, **summarise(x, "year")})
            if y == ts.year:
                w = x[(x.SVC >= ts) & (x.SVC < te)]
                test_rows.append({"mode": m, **summarise(w, "test window")})
            if m == "keep":                                   # 5 visits whose only PA lines are quality codes
                pv = x[x.y == 1].groupby("CSN").q.agg(["all", "size"])
                only = pv[pv["all"]]
                flips.append({"YEAR": y, "PA_visits": len(pv), "visits_only_quality_PA": len(only),
                              "share": len(only) / max(len(pv), 1)})
        print(f"[{y}] lines {len(t):,} | quality {len(q):,} ({len(q) / len(t):.1%}) | distinct codes {q.FINAL_CPT.nunique()}")
        del t, q

    C = pd.concat(codes, ignore_index=True)
    tot = C.groupby("FINAL_CPT")[["lines", "TRUE", "FALSE", "AMBIGUOUS"]].sum().sort_values("lines", ascending=False)
    tot["TRUE_rate_labelled"] = (tot.TRUE / (tot.TRUE + tot.FALSE).replace(0, np.nan)).round(4)
    tot.to_csv(f"{a.out}/1_distinct_codes.csv")
    C.to_csv(f"{a.out}/1_distinct_codes_by_year.csv", index=False)
    R = pd.DataFrame(raw); R.to_csv(f"{a.out}/2_raw_counts.csv", index=False)
    M = pd.DataFrame(modes); M.to_csv(f"{a.out}/3_modes_by_year.csv", index=False)
    T = pd.DataFrame(test_rows); T.to_csv(f"{a.out}/4_test_window_modes.csv", index=False)
    F = pd.DataFrame(flips); F.to_csv(f"{a.out}/5_visits_flipped.csv", index=False)

    print(f"\n== 1. distinct quality codes across all years: {len(tot)} | top 25 by lines\n", tot.head(25).to_string())
    print("\n== 2. raw file (before any cleaning)\n", R.round(3).to_string(index=False))
    show = ["YEAR", "mode", "lines", "PA_lines", "PA_line_rate", "services", "PA_services", "PA_service_rate",
            "visits", "PA_visits", "PA_visit_rate"]
    print("\n== 3. pipeline data by mode (AMBIGUOUS dropped, dedup keep TRUE)\n", M[show].round(4).to_string(index=False))
    allm = M.groupby("mode")[["lines", "PA_lines", "services", "PA_services", "visits", "PA_visits"]].sum()
    for k in ["line", "service", "visit"]:
        allm[f"PA_{k}_rate"] = allm[f"PA_{k}s"] / allm[f"{k}s"]
    print("\n== 3b. all years together\n", allm.round(4).to_string())
    print("\n== 4. test window by mode\n", T[[c for c in show if c != "YEAR"]].round(4).to_string(index=False))
    print("\n== 5. PA visits whose only PA lines are quality codes (visit label lost if quality -> FALSE)\n",
          F.round(4).to_string(index=False))
    print(f"\noutputs in {a.out}/")


if __name__ == "__main__":
    main()
