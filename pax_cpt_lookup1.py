"""
pax_cpt_lookup.py - answer "what did the passenger fix do to CPT X?" from the data, per payer.

  python pax_cpt_lookup.py --build              # once (a few minutes): reads the v8-draft files, saves pax_cpt_payer_summary.parquet
  python pax_cpt_lookup.py 11730 99214          # instant: one table per CPT, a row per payer + a total row
  python pax_cpt_lookup.py 11730 --payer "UNITED HEALTHCARE MEDICAID"
  python pax_cpt_lookup.py --risky              # every CPT + payer with flips but no / weak / conflicting evidence at that payer

Per CPT + payer (quality codes excluded, all years 2022-2026, labels LABEL_V6 -> LABEL_V6_PAX):
  lines          all lines
  pa_before      PA lines before the fix         pa_after   PA lines after      flipped   lines relabelled (IS_PASSENGER)
  vla_kept       lines kept PA as visit-level auth (no owner on the claim)
  alone          lines billed alone on their claim (only CPT on the claim)    alone_pa   of those, PA
  alone_pa_rate  alone_pa / alone
  evidence       for combinations with flips: does THIS payer's own alone rate agree with the rule (< 2%)?
                   CONFLICT  alone PA rate at this payer >= 2% (the rule's threshold): the payer's own data disagrees
                   good      >= 10 alone lines, alone PA rate < 2%     moderate  3-9 alone lines, rate < 2%
                   weak      1-2 alone lines, none PA                  none      never billed alone at this payer
Claim = CSN + FILING_ORDER_C; auth cleaning as in build_v7_1.py.
"""
import argparse
import os

import numpy as np
import pandas as pd

V8 = "s3://cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v8_draft/ai_prior_auth_09282026_v3_{y}_std.parquet"
OUT = "pax_cpt_payer_summary.parquet"
T = {"TRUE", "1", "1.0"}
THR = 0.02          # the passenger rule's alone-rate threshold
COLS = ["CSN", "FILING_ORDER_C", "FINAL_CPT", "PAYOR", "LABEL_V6", "LABEL_V6_PAX", "IS_PASSENGER", "VISIT_LEVEL_AUTH"]


def build(path, years):
    fs = None
    if path.startswith("s3://"):
        import pyarrow.fs as pafs
        fs, path = pafs.S3FileSystem(), path.replace("s3://", "")
    parts = []
    for y in years:
        d = pd.read_parquet(path.format(y=y), columns=COLS, filesystem=fs)
        d["cpt"] = d.FINAL_CPT.astype(str).str.strip().str.upper()
        d["payor"] = d.PAYOR.astype(str).str.upper().str.replace(r"\bAS OF\b.*$", "", regex=True) \
            .str.replace(r"\s+", " ", regex=True).str.strip()
        claim = d.CSN.astype(str).str.replace(r"\.0$", "", regex=True) + "|" + \
            pd.to_numeric(d.FILING_ORDER_C, errors="coerce").fillna(1).astype(int).astype(str)
        d["alone"] = d.groupby(claim).cpt.transform("nunique").eq(1)          # counted before quality codes are dropped
        d = d[~d.cpt.str.fullmatch(r"\d{4}F")]                                 # quality codes excluded, as in training
        d["pa_before"] = d.LABEL_V6.astype(str).str.strip().str.upper().isin(T)
        d["pa_after"] = d.LABEL_V6_PAX.astype(str).str.strip().str.upper().isin(T)
        d["flipped"] = pd.to_numeric(d.IS_PASSENGER, errors="coerce").fillna(0).eq(1)
        d["vla_kept"] = pd.to_numeric(d.VISIT_LEVEL_AUTH, errors="coerce").fillna(0).eq(1)
        d["alone_pa"] = d.alone & d.pa_before
        parts.append(d.groupby(["cpt", "payor"])[["pa_before", "pa_after", "flipped", "vla_kept", "alone", "alone_pa"]]
                     .sum().assign(lines=d.groupby(["cpt", "payor"]).size()))
        print(y, "done")
    s = pd.concat(parts).groupby(level=[0, 1]).sum().reset_index()
    s.to_parquet(OUT, index=False)
    print(f"saved {OUT}: {len(s):,} CPT + payer rows, {s.cpt.nunique():,} CPTs")


def evidence(s):
    rate = s.alone_pa / s.alone.where(s.alone > 0)
    e = np.select([rate >= THR, s.alone >= 10, s.alone >= 3, s.alone >= 1], ["CONFLICT", "good", "moderate", "weak"], "none")
    return pd.Series(np.where(s.flipped > 0, e, ""), index=s.index)


def show(s):
    s = s.assign(pa_rate_before=s.pa_before / s.lines, alone_pa_rate=s.alone_pa / s.alone.where(s.alone > 0), evidence=evidence(s))
    cols = ["lines", "pa_before", "pa_after", "flipped", "vla_kept", "alone", "alone_pa", "alone_pa_rate", "pa_rate_before", "evidence"]
    return s[["payor"] + cols] if "payor" in s else s[cols]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cpts", nargs="*")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--path", default=V8)
    ap.add_argument("--years", nargs="+", type=int, default=[2022, 2023, 2024, 2025, 2026])
    ap.add_argument("--payer", default=None, help="only payers containing this text")
    ap.add_argument("--risky", action="store_true")
    ap.add_argument("--top", type=int, default=40, help="payer rows shown per CPT (most PA first)")
    a = ap.parse_args()
    if a.build:
        build(a.path, a.years)
    if not (a.cpts or a.risky):
        return
    if not os.path.exists(OUT):
        raise SystemExit(f"{OUT} not found: run with --build first")
    s = pd.read_parquet(OUT)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
    if a.payer:
        s = s[s.payor.str.contains(a.payer.upper(), regex=False)]
    for c in a.cpts:
        x = s[s.cpt == c.strip().upper()].sort_values(["pa_before", "lines"], ascending=False)
        if x.empty:
            print(f"\n== {c}: not found"); continue
        tot = x.drop(columns=["cpt", "payor"]).sum().to_frame().T.assign(payor="ALL PAYERS")
        print(f"\n== {c}: {len(x):,} payers | lines {int(tot.lines[0]):,} | PA {int(tot.pa_before[0]):,} -> {int(tot.pa_after[0]):,} "
              f"(flipped {int(tot.flipped[0]):,}) | billed alone {int(tot.alone[0]):,}, of which PA {int(tot.alone_pa[0]):,} "
              f"({tot.alone_pa[0] / max(tot.alone[0], 1):.1%})")
        print(show(pd.concat([x.head(a.top), tot], ignore_index=True)).round(3).to_string(index=False))
    if a.risky:
        r = show(s).assign(cpt=s.cpt)
        r = r[r.evidence.isin(["none", "weak", "CONFLICT"])].sort_values("flipped", ascending=False)
        print(f"\n== CPT + payer with flipped lines but no / weak / conflicting evidence at that payer: {len(r):,} rows, "
              f"{int(r.flipped.sum()):,} flipped lines ({int((r.pa_after == 0).sum()):,} rows now at zero PA)")
        print(r.groupby("evidence").agg(rows=("lines", "size"), flipped=("flipped", "sum"),
                                        now_zero_pa=("pa_after", lambda v: int((v == 0).sum()))).to_string())
        a_ = show(s)
        a_ = a_[a_.evidence != ""]
        print(f"\n   all CPT + payer with flips: {len(a_):,} rows, {int(a_.flipped.sum()):,} flipped lines, by evidence:")
        print(a_.groupby("evidence").agg(rows=("lines", "size"), flipped=("flipped", "sum")).to_string())
        print(r[["cpt"] + [c for c in r.columns if c != "cpt"]].head(a.top).round(3).to_string(index=False))
        r.to_csv("pax_risky_combos.csv", index=False)
        print("saved pax_risky_combos.csv")


if __name__ == "__main__":
    main()
