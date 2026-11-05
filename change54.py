"""
derive_test_set.py - rebuild the benchmark test set (no-filter baseline, exp_nofilter_2022) step by step,
standalone (pandas + pyarrow only), printing the counts after every step so they can be compared.

Benchmark test set (expected at the end):
  2,182,493 lines | 835,591 visits (CSN) | 1,949,656 services (CSN + CPT) | 52,278 PA services | PA line rate 2.6768%

Steps (same order as the pipeline, pa_base_adapter.prep + pa_experiments.split):
  0. read the std v4 file of each year (all rows)
  1. label: GROUP_LABEL TRUE/1 -> 1, FALSE/0 -> 0, anything else (AMBIGUOUS, blank) -> dropped
  2. filing order: NO filter - primary AND secondary/tertiary claims kept as their own rows
  3. repeated lines: same CSN + FINAL_CPT + FINAL_MODIFIERS + SERVICEDATE + FILING_ORDER_C -> keep one,
     the TRUE copy if any copy is TRUE (done per year file)
  4. SERVICEDATE parsed; unreadable dates dropped
  5. window: each visit (CSN) goes to the window of its FIRST service date, taken from its primary rows
     (FILING_ORDER_C = 1; all rows if it has none) - computed on the cleaned data of ALL loaded years.
     Test = lines dated 2026-05-01 <= date < 2026-08-01 whose visit's first date is also in that window.
     -> a visit that started in April is NOT in test, even for its May lines.
Services = CSN + FINAL_CPT: label = max over its lines (PA if any line is PA).

Run: python derive_test_set.py --years 2025 2026 --out test_set_rebuilt.parquet
     (2025 is loaded only so visits that started earlier are placed correctly; 2022-2026 gives the same test set)
"""
import argparse

import pandas as pd
import pyarrow.fs as pafs
import pyarrow.parquet as pq

PATH = "cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v4/ai_prior_auth_09282026_v3_{year}_std.parquet"
COLS = ["CSN", "FINAL_CPT", "FINAL_MODIFIERS", "PLACE_OF_SERVICE_CODE", "SERVICEDATE", "FILING_ORDER_C",
        "GROUP_LABEL", "PAYOR", "PLAN", "GROUP_GUID"]
TEST_START, TEST_END = pd.Timestamp("2026-05-01"), pd.Timestamp("2026-08-01")


def show(step, d):
    pa = d[d.y == 1] if "y" in d else d.iloc[:0]
    print(f"{step:<55} lines {len(d):>11,} | visits {d.CSN.nunique():>10,}"
          + (f" | PA lines {len(pa):>8,} | PA visits {pa.CSN.nunique():>7,}" if "y" in d else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, default=[2025, 2026])
    ap.add_argument("--out", default="test_set_rebuilt.parquet")
    a = ap.parse_args()
    fs = pafs.S3FileSystem()
    parts = []
    for y in a.years:
        names = pq.read_schema(PATH.format(year=y), filesystem=fs).names
        d = pd.read_parquet(PATH.format(year=y), filesystem=fs, columns=list(dict.fromkeys(c for c in COLS if c in names)))
        d = d.loc[:, ~d.columns.duplicated()]
        show(f"[{y}] 0. raw file", d)
        lab = d.GROUP_LABEL.astype(str).str.strip().str.upper()
        d["y"] = lab.map({"TRUE": 1, "1": 1, "1.0": 1, "FALSE": 0, "0": 0, "0.0": 0})
        print("      labels:", lab.value_counts().to_dict())
        d = d[d.y.notna()].copy()
        show(f"[{y}] 1. AMBIGUOUS / missing label dropped", d)
        d["FO"] = pd.to_numeric(d.FILING_ORDER_C, errors="coerce")
        print("      filing order:", d.FO.value_counts(dropna=False).sort_index().to_dict(), "(all kept)")
        d = d.sort_values("y", ascending=False, kind="stable").drop_duplicates(
            ["CSN", "FINAL_CPT", "FINAL_MODIFIERS", "SERVICEDATE", "FILING_ORDER_C"]).sort_index()
        show(f"[{y}] 3. repeated lines dropped (keep TRUE copy)", d)
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    d["SVC_DATE"] = pd.to_datetime(d.SERVICEDATE, errors="coerce")
    d = d[d.SVC_DATE.notna()]
    show("4. unreadable service date dropped (all years)", d)

    first = d.groupby("CSN").SVC_DATE.min()
    first_p = d[d.FO == 1].groupby("CSN").SVC_DATE.min()
    first = first_p.reindex(first.index).fillna(first)                   # primary rows first, else any row
    test_visits = first[(first >= TEST_START) & (first < TEST_END)].index
    t = d[(d.SVC_DATE >= TEST_START) & (d.SVC_DATE < TEST_END)]
    show("5a. lines dated in the test window", t)
    t = t[t.CSN.isin(test_visits)]
    show("5b. ...and visit's first date in the test window = TEST", t)

    s = t.groupby(["CSN", "FINAL_CPT"]).y.max()
    print(f"\nTEST: {len(t):,} lines | {t.CSN.nunique():,} visits | {len(s):,} services | "
          f"{int(s.sum()):,} PA services | PA line rate {t.y.mean():.4%}")
    print("expected: 2,182,493 lines | 835,591 visits | 1,949,656 services | 52,278 PA services | 2.6768%")
    t.drop(columns=["FO"]).to_parquet(a.out, index=False)
    print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()
