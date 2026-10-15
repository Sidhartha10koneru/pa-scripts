"""
build_eda_charts.py - PA-vs-No-PA data readiness EDA (v2).

Question this answers: "Is the claims data ready to train the PA model?"

Processes one year at a time, reduces each year to small count tables, discards the
row-level frame, and accumulates across years (no more than one year in memory - the
fix for the earlier OOM kill). Data only: no rules file / master sheet is used.

Charts (outdir):
  00_readiness_scorecard   one page: pass / warn / fail per readiness question (+ .csv)
  01_funnel                how many lines carry a usable label
  02_payor_rate            where PA concentrates
  03_feature_association   bias-corrected Cramer's V per feature (fixes high-cardinality bias)
  04_monthly_stability     volume and PA rate per month, all years
  05_cell_evidence         (replaces volume gate) labelled evidence and positives per CPT+PAYOR+PLAN cell
  06_cell_purity           (replaces decidability) do CPT+PAYOR+PLAN cells alone separate PA?
  07_label_by_year         does the label mean the same thing every year?
  08_label_noise           share of TRUE lines that are companion codes (office visits, labs...)
  09_filing_order          PA rate by payer order (primary vs secondary)
  10_drift                 payor and CPT mix of each year vs the latest year (PSI)
  11_missingness           null % per feature per year, and TRUE vs FALSE
  12_claim_structure       lines per claim, duplicates, claims with mixed labels
  13_coverage              latest-year lines whose CPT / payor / cell were seen in earlier years

Run (terminal or !python in a notebook):
    python build_eda_charts.py --outdir ./eda_charts_all_years
    python build_eda_charts.py --years 2026 --outdir ./eda_charts_2026
"""
import argparse
import gc
import json
import logging
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.fs as pafs
from scipy.stats import chi2_contingency

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

YEARS = [2022, 2023, 2024, 2025, 2026]
BASE_PATH = "cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v4/ai_prior_auth_09282026_v3_{year}_std.parquet"

NEEDED_COLS = [
    "GROUP_LABEL", "PAYOR", "PLAN", "FINAL_CPT", "SERVICEDATE", "SPECIALTY",
    "VISITTYPE", "SERVICEPROVIDERNPI", "DEPARTMENT", "Primary_DX",
    "CSN", "FILING_ORDER_C",                      # new: claim structure + payer order
]
# Every large-string column we touch must be downcast - this was the missing
# piece (Primary_DX, SERVICEPROVIDERNPI) that caused the OOM kill.
CATEGORY_COLS = ["GROUP_LABEL", "PAYOR", "PLAN", "FINAL_CPT", "SPECIALTY",
                 "VISITTYPE", "DEPARTMENT", "Primary_DX", "SERVICEPROVIDERNPI", "CSN"]
CELL_COLS = ["FINAL_CPT", "PAYOR", "PLAN"]
FEATURE_COLS = ["PLAN", "FINAL_CPT", "PAYOR", "VISITTYPE", "SERVICEPROVIDERNPI",
                "DEPARTMENT", "Primary_DX", "SPECIALTY"]
MISSING_COLS = FEATURE_COLS + ["SERVICEDATE", "FILING_ORDER_C"]
LABELS = ["TRUE", "FALSE", "AMBIGUOUS"]

# Values that are really "missing" even though they are not NaN (we were caught by "nan" strings before)
NULL_TOKENS = {"", "NAN", "NONE", "NULL", "NA", "N/A", "UNKNOWN", "UNK", "?", "-", "0", "0000000000"}
# Lines that ride along on a PA claim but never need PA themselves
COMPANION = re.compile(r"^(99[2-4]\d\d|\d{4}F|36415|96[3-5]\d\d|Q996[5-7]|A957[5-9]|G2211)$")
MIN_EVIDENCE = 20           # labelled (TRUE+FALSE) lines before a cell's PA rate is trusted

COLOR_GRAY, COLOR_NAVY = "#8c96ab", "#1b2a4a"
COLOR_RED, COLOR_GREEN = "#c0504d", "#4f9d69"
COLOR_ORANGE, COLOR_BLUE = "#e08e2b", "#3a5fcd"
plt.rcParams.update({"font.size": 12, "axes.titleweight": "bold", "axes.titlesize": 15})


# ============================================================== helpers
def cell_label_counts(df, group_cols):
    """Safe replacement for groupby(...).value_counts() - that pattern ignores
    observed=True on the nested categorical column and enumerates every possible
    category combination, which caused the earlier OOM kill."""
    is_true = (df["GROUP_LABEL"] == "TRUE").astype("int32")
    is_false = (df["GROUP_LABEL"] == "FALSE").astype("int32")
    is_ambiguous = (df["GROUP_LABEL"] == "AMBIGUOUS").astype("int32")
    counts = pd.DataFrame({"TRUE": is_true, "FALSE": is_false, "AMBIGUOUS": is_ambiguous})
    return counts.groupby([df[c] for c in group_cols], observed=True, sort=False).sum()


def _as_str_index(obj):
    """Plain-string index/columns so tables from different years add up cleanly
    (each year has its own category set)."""
    obj = obj.copy()
    if isinstance(obj.index, pd.MultiIndex):
        obj.index = pd.MultiIndex.from_arrays(
            [obj.index.get_level_values(i).astype(str) for i in range(obj.index.nlevels)],
            names=obj.index.names)
    else:
        obj.index = obj.index.astype(str)
    if isinstance(obj, pd.DataFrame):
        obj.columns = obj.columns.astype(str)
    return obj


def _to_category_with_nulls(s):
    """Downcast to category and turn placeholder values ('nan', 'UNKNOWN', '') into real NaN."""
    s = s.astype("category")
    bad = [c for c in s.cat.categories if str(c).strip().upper() in NULL_TOKENS]
    return s.cat.remove_categories(bad) if bad else s


def _add(a, b):
    return a.add(b, fill_value=0)


def binary_entropy(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def cramers_v_corrected(ct):
    """Bias-corrected Cramer's V (Bergsma 2013). Plain V inflates for columns with many
    distinct values (NPI, CPT, DX), so those always look 'strongest' without the correction."""
    ct = ct.loc[ct.sum(axis=1) > 0, ct.sum(axis=0) > 0]
    r, k = ct.shape
    if r < 2 or k < 2:
        return 0.0
    n = ct.values.sum()
    chi2 = chi2_contingency(ct, correction=False)[0]
    phi2 = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    denom = min(kc - 1, rc - 1)
    return float(np.sqrt(phi2 / denom)) if denom > 0 else 0.0


def psi(p, q, eps=1e-6):
    """Population stability index between two share distributions (index = category)."""
    idx = p.index.union(q.index)
    p = p.reindex(idx).fillna(0) + eps
    q = q.reindex(idx).fillna(0) + eps
    p, q = p / p.sum(), q / q.sum()
    return float(((p - q) * np.log(p / q)).sum())


def _save(fig, outdir, name):
    fig.tight_layout()
    fig.savefig(f"{outdir}/{name}.png", dpi=150)
    plt.close(fig)
    logger.info("Saved %s.png", name)


# ============================================================== per-year reduction
def process_year(year, filesystem, base_path=BASE_PATH, specialty=None):
    path = base_path.format(year=year)
    logger.info("Loading %s", path)
    df = pd.read_parquet(path, columns=NEEDED_COLS, filesystem=filesystem)
    logger.info("  %d rows loaded for %d", len(df), year)

    if specialty:
        df = df[df["SPECIALTY"] == specialty]

    for col in CATEGORY_COLS:
        if col in df.columns:
            df[col] = _to_category_with_nulls(df[col])
    df["SERVICEDATE"] = pd.to_datetime(df["SERVICEDATE"], errors="coerce")
    df["month"] = df["SERVICEDATE"].dt.to_period("M")
    df["FILING_ORDER_C"] = pd.to_numeric(df["FILING_ORDER_C"], errors="coerce")

    funnel_counts = df["GROUP_LABEL"].value_counts()          # single Series, not grouped
    usable = df[df["GROUP_LABEL"].isin(["TRUE", "FALSE"])]

    payor_counts = cell_label_counts(usable, ["PAYOR"])
    cell_counts = cell_label_counts(df, CELL_COLS)
    monthly_counts = cell_label_counts(usable, ["month"])
    feature_crosstabs = {col: pd.crosstab(usable[col], usable["GROUP_LABEL"])
                         for col in FEATURE_COLS if col in usable.columns}

    # ---- label noise: companion codes among TRUE / FALSE
    cats = df["FINAL_CPT"].cat.categories
    comp_codes = [c for c in cats if COMPANION.match(str(c).strip().upper())]
    is_comp = usable["FINAL_CPT"].isin(comp_codes).map({True: "companion", False: "other"})
    companion_ct = pd.crosstab(is_comp, usable["GROUP_LABEL"])

    # ---- payer order
    filing_ct = pd.crosstab(df["FILING_ORDER_C"].fillna(-1).astype(int).astype(str), df["GROUP_LABEL"])

    # ---- missingness: null counts per feature, split by label
    miss_cols = [c for c in MISSING_COLS if c in df.columns]
    null_by_label = df[miss_cols].isna().groupby(df["GROUP_LABEL"], observed=True).sum()
    rows_by_label = df["GROUP_LABEL"].value_counts()

    # ---- claim structure (CSN = claim)
    claim = {}
    if "CSN" in df.columns:
        sizes = df.groupby("CSN", observed=True).size()
        claim["n_lines"] = int(len(df))
        claim["n_claims"] = int(len(sizes))
        claim["lines_per_claim_hist"] = sizes.clip(upper=20).value_counts().sort_index()
        claim["dup_lines"] = int(df.duplicated(["CSN", "FINAL_CPT", "SERVICEDATE"]).sum())
        lab = usable.groupby("CSN", observed=True)["GROUP_LABEL"].nunique()
        claim["claims_labelled"] = int((lab > 0).sum())
        claim["claims_mixed_label"] = int((lab > 1).sum())

    del df, usable
    gc.collect()
    return {
        "year": year,
        "funnel_counts": _as_str_index(funnel_counts),
        "payor_counts": _as_str_index(payor_counts),
        "cell_counts": _as_str_index(cell_counts),
        "monthly_counts": _as_str_index(monthly_counts),
        "feature_crosstabs": {c: _as_str_index(ct) for c, ct in feature_crosstabs.items()},
        "companion_ct": _as_str_index(companion_ct),
        "filing_ct": _as_str_index(filing_ct),
        "null_by_label": _as_str_index(null_by_label),
        "rows_by_label": _as_str_index(rows_by_label),
        "claim": claim,
    }


def accumulate(years, filesystem, base_path=BASE_PATH, specialty=None):
    """Keeps each year's small tables (for year-vs-year checks) and their sum (all years)."""
    per_year = [process_year(y, filesystem, base_path, specialty) for y in years]
    total = {}
    for key in ["funnel_counts", "payor_counts", "cell_counts", "monthly_counts",
                "companion_ct", "filing_ct", "null_by_label", "rows_by_label"]:
        acc = per_year[0][key]
        for ya in per_year[1:]:
            acc = _add(acc, ya[key])
        total[key] = acc
    total["feature_crosstabs"] = {}
    for col in per_year[0]["feature_crosstabs"]:
        acc = per_year[0]["feature_crosstabs"][col]
        for ya in per_year[1:]:
            acc = _add(acc, ya["feature_crosstabs"][col])
        total["feature_crosstabs"][col] = acc.fillna(0)
    logger.info("Accumulated all years: %d total GROUP_LABEL rows", int(total["funnel_counts"].sum()))
    return total, per_year


def _tf(t):
    """TRUE / FALSE columns (or index entries) with 0 when absent."""
    get = t.get if hasattr(t, "get") else (lambda k, d: d)
    return get("TRUE", 0), get("FALSE", 0)


# ============================================================== existing charts (kept)
def plot_funnel(funnel_counts, outdir):
    total_lines = int(funnel_counts.sum())
    n_true, n_false = int(funnel_counts.get("TRUE", 0)), int(funnel_counts.get("FALSE", 0))
    n_usable = n_true + n_false
    pct_usable = 100 * n_usable / total_lines
    funnel = pd.DataFrame({
        "stage": [f"All claim lines\n{total_lines:,}", f"Usable label\n(TRUE+FALSE) {n_usable:,}",
                  f"PA required\n(TRUE) {n_true:,}", f"PA not required\n(FALSE) {n_false:,}"],
        "value": [total_lines, n_usable, n_true, n_false],
        "color": [COLOR_GRAY, COLOR_NAVY, COLOR_RED, COLOR_GREEN],
    })
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(funnel["stage"], funnel["value"], color=funnel["color"])
    ax.set_xscale("log"); ax.set_xlabel("claim lines (log scale)")
    ax.set_title(f"Only ~{pct_usable:.1f}% of {total_lines/1e6:.2f}M lines yield a usable label")
    ax.invert_yaxis()
    for i, v in enumerate(funnel["value"]):
        ax.text(v * 1.05, i, f"{v:,.0f}", va="center", fontweight="bold", color=COLOR_NAVY)
    _save(fig, outdir, "01_funnel")
    return {"total_lines": total_lines, "usable_lines": n_usable, "true_lines": n_true,
            "usable_pct": pct_usable, "true_rate_pct": 100 * n_true / max(n_usable, 1)}


def plot_payor_rate(payor_counts, outdir, top_n=10, min_cell_total=100):
    pc = payor_counts.copy()
    pc["total"] = pc.get("TRUE", 0) + pc.get("FALSE", 0)
    base_rate = 100 * pc.get("TRUE", 0).sum() / pc["total"].sum()
    eligible = pc[pc["total"] >= min_cell_total].assign(pa_rate_pct=lambda d: 100 * d.get("TRUE", 0) / d["total"])
    top = eligible.sort_values("pa_rate_pct", ascending=False).head(top_n)
    colors = [COLOR_RED if r > base_rate else COLOR_ORANGE for r in top["pa_rate_pct"]]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top.index[::-1], top["pa_rate_pct"][::-1], color=colors[::-1])
    ax.axvline(base_rate, color=COLOR_NAVY, linestyle="--", linewidth=1)
    ax.text(base_rate, len(top) - 0.3, f"base {base_rate:.1f}%", ha="center", fontsize=10)
    ax.set_xlabel("PA-required rate (%)"); ax.set_title("PA requirement concentrates in specific payors")
    for i, v in enumerate(top["pa_rate_pct"][::-1]):
        ax.text(v + 0.1, i, f"{v:.1f}%", va="center", fontweight="bold", color=COLOR_NAVY)
    _save(fig, outdir, "02_payor_rate")
    # concentration: share of TRUE lines in the top 10 payors by TRUE volume
    t = pc.get("TRUE", pd.Series(dtype=float)).sort_values(ascending=False)
    return {"true_share_top10_payors_pct": 100 * t.head(10).sum() / max(t.sum(), 1)}


def plot_feature_association(feature_crosstabs, outdir):
    assoc = pd.Series({col: cramers_v_corrected(ct) for col, ct in feature_crosstabs.items()}).sort_values(ascending=False)
    ncat = {col: int((ct.sum(axis=1) > 0).sum()) for col, ct in feature_crosstabs.items()}
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(assoc.index[::-1], assoc.values[::-1], color=COLOR_BLUE)
    ax.set_xlabel("bias-corrected Cramér's V (association with PA-required)")
    ax.set_title("Which features best predict PA requirement?")
    for i, (col, v) in enumerate(assoc[::-1].items()):
        ax.text(v + 0.002, i, f"{v:.3f}  ({ncat[col]:,} values)", va="center", fontweight="bold", color=COLOR_NAVY)
    _save(fig, outdir, "03_feature_association")
    return {"max_cramers_v": float(assoc.max()), "best_feature": assoc.idxmax(), "cramers_v": assoc.round(4).to_dict()}


def plot_monthly_stability(monthly_counts, outdir):
    m = monthly_counts.copy()
    m = m[m.index != "NaT"]
    m["volume"] = m.get("TRUE", 0) + m.get("FALSE", 0)
    m["pa_rate_pct"] = 100 * m.get("TRUE", 0) / m["volume"]
    m = m.sort_index()
    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax1.bar(m.index, m["volume"], color=COLOR_GRAY, alpha=0.6)
    ax1.set_ylabel("supervised lines", color=COLOR_GRAY)
    ax1.set_xticks(range(0, len(m), 3)); ax1.set_xticklabels(m.index[::3], rotation=90)
    ax2 = ax1.twinx()
    ax2.plot(range(len(m)), m["pa_rate_pct"], color=COLOR_RED, marker="o", linewidth=2)
    ax2.set_ylabel("PA-required rate (%)", color=COLOR_RED)
    ax1.set_title(f"Label rate ranges {m['pa_rate_pct'].min():.2f}%-{m['pa_rate_pct'].max():.2f}% by month")
    _save(fig, outdir, "04_monthly_stability")
    mr = m.loc[m["volume"] >= 1000, "pa_rate_pct"]
    return {"monthly_rate_max_over_min": float(mr.max() / max(mr.min(), 1e-9)) if len(mr) else None}


# ============================================================== replaced charts
def plot_cell_evidence(cell_counts, outdir):
    """Replaces the volume gate. The gate bucketed cells by TOTAL lines (incl. AMBIGUOUS) and
    showed % of CELLS. For learning PA what matters is labelled lines and, above all, how many
    TRUE lines a cell has; and the share of VOLUME / TRUE lines, not of cells (most cells are tiny)."""
    c = cell_counts.copy()
    c["labelled"] = c.get("TRUE", 0) + c.get("FALSE", 0)
    t = c.get("TRUE", 0)

    def bucket(row_lab, row_t):
        if row_lab == 0:
            return "1 no labelled lines"
        if row_lab < MIN_EVIDENCE:
            return f"2 <{MIN_EVIDENCE} labelled"
        if row_t == 0:
            return f"3 >={MIN_EVIDENCE} labelled, 0 TRUE"
        if row_t < 5:
            return f"4 >={MIN_EVIDENCE} labelled, 1-4 TRUE"
        return f"5 >={MIN_EVIDENCE} labelled, 5+ TRUE"

    c["bucket"] = [bucket(a, b) for a, b in zip(c["labelled"], t)]
    order = ["1 no labelled lines", f"2 <{MIN_EVIDENCE} labelled", f"3 >={MIN_EVIDENCE} labelled, 0 TRUE",
             f"4 >={MIN_EVIDENCE} labelled, 1-4 TRUE", f"5 >={MIN_EVIDENCE} labelled, 5+ TRUE"]   # always all 5 buckets
    pct_cells = 100 * c["bucket"].value_counts(normalize=True).reindex(order).fillna(0)
    pct_lab = 100 * c.groupby("bucket")["labelled"].sum().reindex(order).fillna(0) / max(c["labelled"].sum(), 1)
    pct_true = 100 * c.groupby("bucket")["TRUE"].sum().reindex(order).fillna(0) / max(t.sum(), 1)

    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(13, 6))
    for off, vals, lab, col in [(-0.27, pct_cells, "% of cells", COLOR_GRAY),
                                (0, pct_lab, "% of labelled lines", COLOR_BLUE),
                                (0.27, pct_true, "% of TRUE lines", COLOR_RED)]:
        ax.bar(x + off, vals, width=0.27, label=lab, color=col)
        for xi, v in zip(x + off, vals):
            ax.text(xi, v + 0.5, f"{v:.0f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([o[2:] for o in order], rotation=15)
    ax.set_ylabel("%"); ax.legend()
    rich = pct_true.get(f"5 >={MIN_EVIDENCE} labelled, 5+ TRUE", 0)
    ax.set_title(f"{rich:.0f}% of TRUE lines sit in CPT+PAYOR+PLAN cells with 5+ TRUE examples")
    _save(fig, outdir, "05_cell_evidence")
    return {"true_in_cells_with_5plus_true_pct": float(rich), "cells": int(len(c)),
            "labelled_in_small_cells_pct": float(pct_lab.get(f"2 <{MIN_EVIDENCE} labelled", 0))}


def plot_cell_purity(cell_counts, outdir):
    """Replaces decidability. The old chart counted every cell, including 1-line cells
    (entropy 0 -> 'decisive' by construction) and the many all-FALSE cells (trivially
    'decisive' at a ~0.4% PA rate). Here: cells with enough evidence only, weighted by
    TRUE lines - i.e. 'does CPT+PAYOR+PLAN alone tell PA from no-PA?'"""
    c = cell_counts.copy()
    c["labelled"] = c.get("TRUE", 0) + c.get("FALSE", 0)
    c = c[c["labelled"] >= MIN_EVIDENCE]
    c["p_true"] = c.get("TRUE", 0) / c["labelled"]
    bins = [-0.001, 0.1, 0.5, 0.9, 1.0]
    names = ["p<0.1 (mostly no-PA)", "0.1-0.5 (mixed)", "0.5-0.9 (mostly PA)", ">=0.9 (PA)"]
    c["band"] = pd.cut(c["p_true"], bins=bins, labels=names)
    true_share = 100 * c.groupby("band", observed=False)["TRUE"].sum() / max(c["TRUE"].sum(), 1)
    lab_share = 100 * c.groupby("band", observed=False)["labelled"].sum() / max(c["labelled"].sum(), 1)

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - 0.2, lab_share.values, width=0.4, label="% of labelled lines", color=COLOR_BLUE)
    ax.bar(x + 0.2, true_share.values, width=0.4, label="% of TRUE lines", color=COLOR_RED)
    for xi, v in zip(x + 0.2, true_share.values):
        ax.text(xi, v + 0.5, f"{v:.0f}%", ha="center", fontweight="bold", color=COLOR_NAVY)
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("%"); ax.legend()
    pure = true_share[["0.5-0.9 (mostly PA)", ">=0.9 (PA)"]].sum()
    ax.set_title(f"{pure:.0f}% of TRUE lines are in cells where PA is the majority "
                 f"(cells with >={MIN_EVIDENCE} labelled lines)")
    _save(fig, outdir, "06_cell_purity")
    return {"true_in_majority_pa_cells_pct": float(pure),
            "mean_entropy_true_weighted": float(np.average(binary_entropy(c["p_true"]), weights=c["TRUE"]))
            if c["TRUE"].sum() else None}


# ============================================================== new readiness charts
def plot_label_by_year(per_year, outdir):
    t = pd.DataFrame({ya["year"]: ya["funnel_counts"] for ya in per_year}).T.reindex(columns=LABELS).fillna(0)
    share = 100 * t.div(t.sum(axis=1), axis=0)
    rate = 100 * t["TRUE"] / (t["TRUE"] + t["FALSE"]).replace(0, np.nan)
    fig, ax = plt.subplots(figsize=(10, 5))
    share.plot(kind="bar", stacked=True, ax=ax, color=[COLOR_RED, COLOR_GREEN, COLOR_GRAY], width=0.6)
    ax.set_ylabel("% of claim lines"); ax.set_xticklabels(share.index, rotation=0)
    ax2 = ax.twinx(); ax2.plot(range(len(rate)), rate.values, color=COLOR_NAVY, marker="o", linewidth=2)
    ax2.set_ylabel("PA rate % (TRUE / (TRUE+FALSE))")
    for i, v in enumerate(rate.values):
        ax2.text(i, v, f" {v:.2f}%", color=COLOR_NAVY, fontweight="bold")
    ax.set_title("Label mix and PA rate by year")
    _save(fig, outdir, "07_label_by_year")
    r = rate.dropna()
    return {"pa_rate_by_year_pct": r.round(3).to_dict(),
            "pa_rate_max_over_min": float(r.max() / max(r.min(), 1e-9)) if len(r) else None,
            "ambiguous_pct_by_year": share["AMBIGUOUS"].round(1).to_dict()}


def plot_label_noise(per_year, outdir):
    rows = {}
    for ya in per_year:
        ct = ya["companion_ct"]
        tr = ct["TRUE"] if "TRUE" in ct else pd.Series(dtype=float)
        rows[ya["year"]] = 100 * tr.get("companion", 0) / max(tr.sum(), 1)
    s = pd.Series(rows)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(s.index.astype(str), s.values, color=COLOR_ORANGE)
    for i, v in enumerate(s.values):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontweight="bold", color=COLOR_NAVY)
    ax.set_ylabel("% of TRUE lines that are companion codes")
    ax.set_title("Label noise: TRUE copied onto office visits, labs, admin codes")
    _save(fig, outdir, "08_label_noise")
    return {"companion_share_of_true_pct": float(s.mean()), "by_year": s.round(1).to_dict()}


def plot_filing_order(filing_ct, outdir):
    f = filing_ct.copy()
    f["usable"] = f.get("TRUE", 0) + f.get("FALSE", 0)
    f = f[f["usable"] > 0]
    f["pa_rate_pct"] = 100 * f.get("TRUE", 0) / f["usable"]
    f["share_pct"] = 100 * f["usable"] / f["usable"].sum()
    f = f.sort_index()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(f.index, f["pa_rate_pct"], color=COLOR_BLUE)
    for i, (r, sh) in enumerate(zip(f["pa_rate_pct"], f["share_pct"])):
        ax.text(i, r, f"{r:.2f}%\n({sh:.1f}% of lines)", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("FILING_ORDER_C (1 = primary?; -1 = missing)"); ax.set_ylabel("PA rate %")
    ax.set_title("PA rate by payer order")
    _save(fig, outdir, "09_filing_order")
    p1 = f.loc["1", "pa_rate_pct"] if "1" in f.index else np.nan
    p2 = f.loc["2", "pa_rate_pct"] if "2" in f.index else np.nan
    return {"pa_rate_primary_pct": float(p1), "pa_rate_secondary_pct": float(p2),
            "secondary_share_pct": float(f["share_pct"].drop("1", errors="ignore").sum())}


def plot_drift(per_year, outdir):
    latest = per_year[-1]
    out = {}
    fig, ax = plt.subplots(figsize=(10, 5))
    for col, colr in [("PAYOR", COLOR_NAVY), ("FINAL_CPT", COLOR_ORANGE)]:
        ref = latest["feature_crosstabs"][col].sum(axis=1)
        ref = ref / ref.sum()
        vals = {}
        for ya in per_year[:-1]:
            cur = ya["feature_crosstabs"][col].sum(axis=1)
            vals[ya["year"]] = psi(cur / cur.sum(), ref)
        s = pd.Series(vals)
        out[col] = s.round(3).to_dict()
        ax.plot(s.index.astype(str), s.values, marker="o", linewidth=2, color=colr, label=col)
    ax.axhline(0.1, color=COLOR_GRAY, linestyle="--"); ax.axhline(0.25, color=COLOR_RED, linestyle="--")
    ax.text(0, 0.105, "0.1 moderate shift", fontsize=9); ax.text(0, 0.255, "0.25 major shift", fontsize=9)
    ax.set_ylabel(f"PSI vs {latest['year']}"); ax.legend()
    ax.set_title(f"How different is each year's payor / CPT mix from {latest['year']}?")
    _save(fig, outdir, "10_drift")
    allv = [v for d in out.values() for v in d.values()]
    return {"psi": out, "max_psi": float(max(allv)) if allv else None}


def plot_missingness(per_year, total, outdir):
    heat = {}
    for ya in per_year:
        nb, rb = ya["null_by_label"], ya["rows_by_label"]
        heat[ya["year"]] = 100 * nb.sum(axis=0) / max(rb.sum(), 1)
    heat = pd.DataFrame(heat)
    nb, rb = total["null_by_label"], total["rows_by_label"]
    by_label = pd.DataFrame({lab: 100 * nb.loc[lab] / rb[lab] for lab in ["TRUE", "FALSE"] if lab in nb.index})

    fig, ax = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [3, 2]})
    im = ax[0].imshow(heat.values, cmap="Reds", vmin=0, vmax=max(5, float(np.nanmax(heat.values))), aspect="auto")
    ax[0].set_xticks(range(heat.shape[1])); ax[0].set_xticklabels(heat.columns)
    ax[0].set_yticks(range(heat.shape[0])); ax[0].set_yticklabels(heat.index)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax[0].text(j, i, f"{heat.values[i, j]:.1f}", ha="center", va="center", fontsize=9)
    ax[0].set_title("% null by year (placeholders like 'nan', 'UNKNOWN' counted as null)")
    fig.colorbar(im, ax=ax[0], fraction=0.03)
    by_label.plot(kind="barh", ax=ax[1], color=[COLOR_RED, COLOR_GREEN])
    ax[1].set_xlabel("% null"); ax[1].set_title("Null % for TRUE vs FALSE lines")
    _save(fig, outdir, "11_missingness")
    gap = (by_label.get("TRUE", 0) - by_label.get("FALSE", 0)).abs() if len(by_label.columns) == 2 else pd.Series(dtype=float)
    return {"max_null_pct": float(np.nanmax(heat.values)), "worst_feature": heat.max(axis=1).idxmax(),
            "max_true_false_null_gap_pp": float(gap.max()) if len(gap) else None,
            "null_pct_all_years": (100 * nb.sum(axis=0) / rb.sum()).round(2).to_dict()}


def plot_claim_structure(per_year, outdir):
    cl = [ya["claim"] for ya in per_year if ya["claim"]]
    if not cl:
        logger.warning("CSN not available - skipping 12_claim_structure")
        return {}
    hist = cl[0]["lines_per_claim_hist"]
    for c in cl[1:]:
        hist = hist.add(c["lines_per_claim_hist"], fill_value=0)
    n_lines = sum(c["n_lines"] for c in cl)
    n_claims = sum(c["n_claims"] for c in cl)
    dup = sum(c["dup_lines"] for c in cl)
    mixed = sum(c["claims_mixed_label"] for c in cl)
    lab = sum(c["claims_labelled"] for c in cl)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(hist.index.astype(int).astype(str), 100 * hist.values / hist.sum(), color=COLOR_NAVY)
    ax.set_xlabel("lines per claim (20 = 20+)"); ax.set_ylabel("% of claims")
    ax.set_title(f"{n_lines/n_claims:.1f} lines per claim | duplicates {100*dup/n_lines:.2f}% of lines | "
                 f"mixed-label claims {100*mixed/max(lab,1):.2f}%", fontsize=12)
    _save(fig, outdir, "12_claim_structure")
    return {"lines_per_claim": n_lines / n_claims, "dup_lines_pct": 100 * dup / n_lines,
            "mixed_label_claims_pct": 100 * mixed / max(lab, 1)}


def plot_coverage(per_year, outdir):
    """Of the latest year's labelled lines, how many have a CPT / payor / cell that earlier
    years saw with >= MIN_EVIDENCE labelled lines? (Can older data teach the newest year?)"""
    if len(per_year) < 2:
        logger.warning("Need 2+ years for 13_coverage - skipped")
        return {}
    latest, prior = per_year[-1], per_year[:-1]
    res = {}
    for name in ["FINAL_CPT", "PAYOR", "cell"]:
        if name == "cell":
            get = lambda ya: ya["cell_counts"][["TRUE", "FALSE"]].reindex(columns=["TRUE", "FALSE"]).fillna(0)
        else:
            get = lambda ya, n=name: ya["feature_crosstabs"][n].reindex(columns=["TRUE", "FALSE"]).fillna(0)
        acc = get(prior[0])
        for ya in prior[1:]:
            acc = _add(acc, get(ya))
        seen = acc.index[acc.sum(axis=1) >= MIN_EVIDENCE]
        cur = get(latest)
        hit = cur.index.isin(seen)
        res[name] = {"labelled_pct": 100 * cur[hit].sum(axis=1).sum() / max(cur.values.sum(), 1),
                     "true_pct": 100 * cur.loc[hit, "TRUE"].sum() / max(cur["TRUE"].sum(), 1)}
    r = pd.DataFrame(res).T
    x = np.arange(len(r))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 0.2, r["labelled_pct"], width=0.4, label="% of labelled lines", color=COLOR_BLUE)
    ax.bar(x + 0.2, r["true_pct"], width=0.4, label="% of TRUE lines", color=COLOR_RED)
    for xi, v in zip(x + 0.2, r["true_pct"]):
        ax.text(xi, v + 1, f"{v:.0f}%", ha="center", fontweight="bold", color=COLOR_NAVY)
    ax.set_xticks(x); ax.set_xticklabels(["CPT", "Payor", "CPT+Payor+Plan"]); ax.set_ylim(0, 110)
    ax.set_ylabel("%"); ax.legend()
    ax.set_title(f"{latest['year']} lines whose value was seen in earlier years (>= {MIN_EVIDENCE} labelled)")
    _save(fig, outdir, "13_coverage")
    return {k: {kk: round(vv, 1) for kk, vv in v.items()} for k, v in res.items()}


# ============================================================== scorecard
def _status(value, good, warn, higher_is_better=True):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    if higher_is_better:
        return "PASS" if value >= good else ("WARN" if value >= warn else "FAIL")
    return "PASS" if value <= good else ("WARN" if value <= warn else "FAIL")


def build_scorecard(m, outdir):
    """Thresholds are starting points to agree with the team, not standards."""
    rows = [
        ("Enough PA examples?", "TRUE lines (all years)", m["funnel"]["true_lines"], *(_status(m["funnel"]["true_lines"], 10000, 1000),), ">=10k pass, >=1k warn"),
        ("Enough usable labels?", "% lines TRUE/FALSE", round(m["funnel"]["usable_pct"], 1), _status(m["funnel"]["usable_pct"], 70, 50), ">=70 pass, >=50 warn"),
        ("Label consistent across years?", "max/min yearly PA rate", m["label_year"]["pa_rate_max_over_min"], _status(m["label_year"]["pa_rate_max_over_min"], 1.5, 3, False), "<=1.5 pass, <=3 warn"),
        ("Label clean?", "% TRUE lines that are companion codes", round(m["noise"]["companion_share_of_true_pct"], 1), _status(m["noise"]["companion_share_of_true_pct"], 10, 30, False), "<=10 pass, <=30 warn"),
        ("Primary vs secondary payer", "PA rate primary / secondary", f'{m["filing"].get("pa_rate_primary_pct", float("nan")):.2f}% / {m["filing"].get("pa_rate_secondary_pct", float("nan")):.2f}%', "INFO", "filter to primary if clearly higher"),
        ("Features carry signal?", "max corrected Cramér's V", round(m["assoc"]["max_cramers_v"], 3), _status(m["assoc"]["max_cramers_v"], 0.1, 0.03), ">=0.1 pass, >=0.03 warn"),
        ("Stable month to month?", "max/min monthly PA rate", m["monthly"]["monthly_rate_max_over_min"], _status(m["monthly"]["monthly_rate_max_over_min"], 2, 4, False), "<=2 pass, <=4 warn"),
        ("Stable year to year?", "max PSI payor/CPT vs latest", m["drift"].get("max_psi"), _status(m["drift"].get("max_psi"), 0.1, 0.25, False), "<=0.1 pass, <=0.25 warn"),
        ("Enough evidence per cell?", "% TRUE lines in cells with 5+ TRUE", round(m["evidence"]["true_in_cells_with_5plus_true_pct"], 1), _status(m["evidence"]["true_in_cells_with_5plus_true_pct"], 70, 40), ">=70 pass, >=40 warn"),
        ("Do cells separate PA?", "% TRUE lines in majority-PA cells", round(m["purity"]["true_in_majority_pa_cells_pct"], 1), "INFO", "low = needs features beyond CPT+payor+plan (or label noise)"),
        ("Features complete?", "worst feature % null", round(m["missing"]["max_null_pct"], 1), _status(m["missing"]["max_null_pct"], 5, 20, False), "<=5 pass, <=20 warn"),
        ("Missingness unrelated to label?", "max |null% TRUE - null% FALSE|", m["missing"]["max_true_false_null_gap_pp"], _status(m["missing"]["max_true_false_null_gap_pp"], 2, 10, False), "<=2pp pass, <=10pp warn"),
        ("No duplicate lines?", "% duplicate lines", round(m["claim"].get("dup_lines_pct", float("nan")), 2), _status(m["claim"].get("dup_lines_pct"), 1, 5, False), "<=1 pass, <=5 warn"),
        ("Label line-level or claim-level?", "% labelled claims with mixed labels", round(m["claim"].get("mixed_label_claims_pct", float("nan")), 2), "INFO", "~0 = label copied per claim -> split train/test by claim"),
        ("Old years teach the newest?", "% latest-year TRUE lines in seen cells", m["coverage"].get("cell", {}).get("true_pct"), _status(m["coverage"].get("cell", {}).get("true_pct"), 80, 50), ">=80 pass, >=50 warn"),
    ]
    sc = pd.DataFrame(rows, columns=["question", "metric", "value", "status", "rule"])
    sc["value"] = sc["value"].map(lambda v: f"{v:,.3f}".rstrip("0").rstrip(".") if isinstance(v, (int, float, np.floating)) and v is not None else v)
    sc.to_csv(f"{outdir}/00_readiness_scorecard.csv", index=False)

    colors = {"PASS": "#d9efdf", "WARN": "#fbe7cc", "FAIL": "#f4d2d1", "INFO": "#e3e8f3", "n/a": "#eeeeee"}
    fig, ax = plt.subplots(figsize=(17, 0.42 * len(sc) + 0.9))
    ax.axis("off")
    tb = ax.table(cellText=sc.values, colLabels=sc.columns, loc="center", cellLoc="left",
                  colWidths=[0.22, 0.27, 0.12, 0.07, 0.32])
    tb.auto_set_font_size(False); tb.set_fontsize(10); tb.scale(1, 1.6)
    ax.margins(0, 0)
    for i, st in enumerate(sc["status"], start=1):
        for j in range(len(sc.columns)):
            tb[i, j].set_facecolor(colors.get(st, "white"))
    for j in range(len(sc.columns)):
        tb[0, j].set_text_props(fontweight="bold"); tb[0, j].set_facecolor("#cfd6e6")
    counts = sc["status"].value_counts()
    ax.set_title(f"Data readiness scorecard - PASS {counts.get('PASS', 0)} | WARN {counts.get('WARN', 0)} | "
                 f"FAIL {counts.get('FAIL', 0)}", fontsize=15, fontweight="bold")
    _save(fig, outdir, "00_readiness_scorecard")
    return sc


# ============================================================== main
def parse_args():
    p = argparse.ArgumentParser(description="PA data readiness EDA across std data years.")
    p.add_argument("--years", nargs="+", type=int, default=YEARS)
    p.add_argument("--specialty", default=None)
    p.add_argument("--outdir", default="./eda_charts")
    p.add_argument("--base-path", default=BASE_PATH)
    return p.parse_known_args()[0]          # ignores Jupyter's own arguments


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    fs = pafs.LocalFileSystem() if args.base_path.startswith("/") else pafs.S3FileSystem()
    total, per_year = accumulate(sorted(args.years), fs, args.base_path, args.specialty)

    m = {}
    m["funnel"] = plot_funnel(total["funnel_counts"], args.outdir)
    m["payor"] = plot_payor_rate(total["payor_counts"], args.outdir)
    m["assoc"] = plot_feature_association(total["feature_crosstabs"], args.outdir)
    m["monthly"] = plot_monthly_stability(total["monthly_counts"], args.outdir)
    m["evidence"] = plot_cell_evidence(total["cell_counts"], args.outdir)
    m["purity"] = plot_cell_purity(total["cell_counts"], args.outdir)
    m["label_year"] = plot_label_by_year(per_year, args.outdir)
    m["noise"] = plot_label_noise(per_year, args.outdir)
    m["filing"] = plot_filing_order(total["filing_ct"], args.outdir)
    m["drift"] = plot_drift(per_year, args.outdir) if len(per_year) > 1 else {}
    m["missing"] = plot_missingness(per_year, total, args.outdir)
    m["claim"] = plot_claim_structure(per_year, args.outdir)
    m["coverage"] = plot_coverage(per_year, args.outdir)
    build_scorecard(m, args.outdir)

    with open(f"{args.outdir}/readiness_metrics.json", "w") as f:
        json.dump(m, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    logger.info("All charts saved to %s", args.outdir)


if __name__ == "__main__":
    main()
