"""
build_eda_charts.py - PA-vs-No-PA data readiness EDA (v4).

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
  05_prd_volume_gates      PRD 7.3 gates on the PRD cell (CPT + modifier + payor/plan + place of service),
                           counted in distinct labelled ENCOUNTERS (CSN), with the 50-example minority-class floor
  06_cell_purity           do PRD cells alone separate PA? (distinct claims, companion codes excluded)
  07_label_by_year         label mix, PA rate, and distinct PA claims per year
  08_label_noise           share of TRUE lines that are companion codes (office visits, labs...)
  09_filing_order          PA rate by payer order (primary vs secondary)
  10_drift                 payor and CPT mix of each year vs the latest year (PSI)
  11_missingness           null % per feature per year, and TRUE vs FALSE
  12_claim_structure       lines per claim, duplicates, claims with mixed labels
  13_coverage              latest-year lines whose CPT / payor / cell were seen in earlier years
  14_baseline              learnability test: PA rates from earlier years predict the latest year (PR curve)
  15_cpt_solo_vs_with      does a CPT carry PA on its own, or only when other codes are on the encounter?
                           (data-based view of 'companion' codes; full table in 15_cpt_solo_vs_with.csv)
  16_field_presence        PA rate when an optional field is filled vs empty (leakage check for POS etc.)
  17_line_of_business      PA rate by line of business (label sanity: self-pay should be ~0)
  REMARKS.md               auto-written observations per chart + a 'Team remark' slot to fill in

Unit: CSN is treated as the ENCOUNTER (label is set per CSN and copied to its lines).
'Companion' codes are a hand-picked, UNVERIFIED list: nothing is excluded by default; results are
shown with all codes, and 'excluding flagged codes' only as a side figure.

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
    "CSN", "FILING_ORDER_C",                      # claim structure + payer order
    "FINAL_MODIFIERS", "PLACE_OF_SERVICE_CODE",   # PRD 7.3 cell definition
    "LINE_OF_BUSINESS", "PRODUCT_NAME",           # optional; skipped if a year's file lacks them
]
OPTIONAL_COLS = {"FINAL_MODIFIERS", "PLACE_OF_SERVICE_CODE", "LINE_OF_BUSINESS", "PRODUCT_NAME", "FILING_ORDER_C"}
# Optional fields: is the field being FILLED itself linked to PA? (if so: possible leakage)
PRESENCE_COLS = ["PLACE_OF_SERVICE_CODE", "FINAL_MODIFIERS", "LINE_OF_BUSINESS", "PRODUCT_NAME", "Primary_DX"]
# Every large-string column we touch must be downcast - this was the missing
# piece (Primary_DX, SERVICEPROVIDERNPI) that caused the OOM kill.
CATEGORY_COLS = ["GROUP_LABEL", "PAYOR", "PLAN", "FINAL_CPT", "SPECIALTY",
                 "VISITTYPE", "DEPARTMENT", "Primary_DX", "SERVICEPROVIDERNPI", "CSN",
                 "FINAL_MODIFIERS", "PLACE_OF_SERVICE_CODE", "LINE_OF_BUSINESS", "PRODUCT_NAME"]
CELL_COLS = ["FINAL_CPT", "PAYOR", "PLAN"]
FEATURE_COLS = ["PLAN", "FINAL_CPT", "PAYOR", "VISITTYPE", "SERVICEPROVIDERNPI",
                "DEPARTMENT", "Primary_DX", "SPECIALTY", "PLACE_OF_SERVICE_CODE", "FINAL_MODIFIERS",
                "LINE_OF_BUSINESS", "PRODUCT_NAME"]
MISSING_COLS = list(dict.fromkeys(FEATURE_COLS + ["SERVICEDATE", "FILING_ORDER_C"]))
# (FINAL_MODIFIERS left out: most lines legitimately have no modifier)

# PRD 7.3: cell = CPT + modifier + plan + place of service. Plan is keyed with its payor because
# claims plan names repeat across payors ("COMMERCIAL"...). Blank modifier / POS -> "NONE".
PRD_KEYS = ["FINAL_CPT", "MOD_KEY", "PAYOR", "PLAN", "POS_KEY"]
GATE_LOW, GATE_HIGH = 100, 500    # <100 rules only | 100-500 ML "required" only | >500 ML both directions
MINORITY_FLOOR = 50               # PRD initial target: min examples of a label in the cell to predict it
LABELS = ["TRUE", "FALSE", "AMBIGUOUS"]

# Values that are really "missing" even though they are not NaN (we were caught by "nan" strings before)
NULL_TOKENS = {"", "NAN", "NONE", "NULL", "NA", "N/A", "UNKNOWN", "UNK", "?", "-", "0", "0000000000"}
# HAND-PICKED, UNVERIFIED list of codes that may ride along on a PA encounter (visits, Cat II, blood
# draw, infusion admin, contrast, G2211). Used only to FLAG, never to drop. See chart 15 for evidence.
COMPANION = re.compile(r"^(99[2-4]\d\d|\d{4}F|36415|96[3-5]\d\d|Q996[5-7]|A957[5-9]|G2211)$")
MIN_EVIDENCE = 20           # labelled claims before a cell's PA rate is used (purity, coverage, baseline backoff)

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


def psi(p, q, top_k=30, eps=1e-4):
    """Population stability index between two share distributions (index = category).
    Categories are binned into the reference's top_k values + 'OTHER'. Without binning,
    thousands of rare CPTs / payor names present in one year but not the other each add
    a large log term and PSI explodes (v2 showed CPT PSI ~5 for that reason)."""
    keep = q.sort_values(ascending=False).index[:top_k]
    def binned(s):
        b = s.reindex(keep).fillna(0)
        b["OTHER"] = s.drop(keep, errors="ignore").sum()
        return b
    p, q = binned(p) + eps, binned(q) + eps
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
    import pyarrow.parquet as pq
    names = set(pq.ParquetFile(filesystem.open_input_file(path)).schema_arrow.names)
    missing = [c for c in NEEDED_COLS if c not in names]
    if [c for c in missing if c not in OPTIONAL_COLS]:
        raise KeyError(f"{year}: required columns missing: {missing}")
    if missing:
        logger.warning("  %d: optional columns not in file, skipped: %s", year, missing)
    df = pd.read_parquet(path, columns=[c for c in NEEDED_COLS if c in names], filesystem=filesystem)
    logger.info("  %d rows loaded for %d", len(df), year)

    if specialty:
        df = df[df["SPECIALTY"] == specialty]

    for col in CATEGORY_COLS:
        if col in df.columns:
            df[col] = _to_category_with_nulls(df[col])
    df["SERVICEDATE"] = pd.to_datetime(df["SERVICEDATE"], errors="coerce")
    df["month"] = df["SERVICEDATE"].dt.to_period("M")
    if "FILING_ORDER_C" not in df.columns:
        df["FILING_ORDER_C"] = np.nan
    df["FILING_ORDER_C"] = pd.to_numeric(df["FILING_ORDER_C"], errors="coerce")

    for src, key in [("FINAL_MODIFIERS", "MOD_KEY"), ("PLACE_OF_SERVICE_CODE", "POS_KEY")]:
        k = df[src] if src in df.columns else pd.Series(pd.NA, index=df.index, dtype="category")
        k = k.astype("category")
        df[key] = k.cat.add_categories(["NONE"]).fillna("NONE") if "NONE" not in k.cat.categories else k.fillna("NONE")
    comp_codes_all = [c for c in df["FINAL_CPT"].cat.categories if COMPANION.match(str(c).strip().upper())]
    df["is_comp"] = df["FINAL_CPT"].isin(comp_codes_all)

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

    # ---- PRD cells in distinct labelled CLAIMS (label is per claim; repeated lines are not new evidence)
    pc = usable[PRD_KEYS + ["CSN", "GROUP_LABEL"]].drop_duplicates(PRD_KEYS + ["CSN"])
    prd_cells = cell_label_counts(pc, PRD_KEYS)[["TRUE", "FALSE"]]
    del pc
    # ---- distinct PA claims (all, and excluding companion-only evidence)
    is_t = usable["GROUP_LABEL"] == "TRUE"
    pa_claims = {"labelled_claims": int(usable["CSN"].nunique()),
                 "pa_claims": int(usable.loc[is_t, "CSN"].nunique()),
                 "pa_claims_non_companion": int(usable.loc[is_t & ~usable["is_comp"], "CSN"].nunique())}

    # ---- CPT on its own vs with other codes (distinct encounter x CPT)
    u = usable[["CSN", "FINAL_CPT", "GROUP_LABEL"]].drop_duplicates(["CSN", "FINAL_CPT"])
    n_cpt = u.groupby("CSN", observed=True)["FINAL_CPT"].transform("size")
    u = u.assign(ctx=np.where(n_cpt.values == 1, "solo", "with_others"))
    solo_ct = cell_label_counts(u, ["FINAL_CPT", "ctx"])[["TRUE", "FALSE"]]
    del u, n_cpt
    # ---- optional fields: PA rate when filled vs empty (lines)
    presence = {}
    for col in PRESENCE_COLS:
        if col in usable.columns:
            filled = usable[col].notna().map({True: "filled", False: "empty"})
            presence[col] = pd.crosstab(filled, usable["GROUP_LABEL"])
    # ---- line of business (encounter level: distinct CSN per LOB)
    lob_ct = None
    if "LINE_OF_BUSINESS" in usable.columns:
        lb = usable[["CSN", "LINE_OF_BUSINESS", "GROUP_LABEL"]].drop_duplicates(["CSN", "LINE_OF_BUSINESS"])
        lob_ct = cell_label_counts(lb, ["LINE_OF_BUSINESS"])[["TRUE", "FALSE"]]
        del lb

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
        claim["dup_lines_with_mod_pos"] = int(df.duplicated(["CSN", "FINAL_CPT", "MOD_KEY", "POS_KEY", "SERVICEDATE"]).sum())
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
        "prd_cells": _as_str_index(prd_cells),
        "solo_ct": _as_str_index(solo_ct),
        "presence": {c: _as_str_index(t) for c, t in presence.items()},
        "lob_ct": None if lob_ct is None else _as_str_index(lob_ct),
        "pa_claims": pa_claims,
    }


def accumulate(years, filesystem, base_path=BASE_PATH, specialty=None):
    """Keeps each year's small tables (for year-vs-year checks) and their sum (all years)."""
    per_year = [process_year(y, filesystem, base_path, specialty) for y in years]
    total = {}
    for key in ["funnel_counts", "payor_counts", "cell_counts", "monthly_counts",
                "companion_ct", "filing_ct", "null_by_label", "rows_by_label", "prd_cells"]:
        acc = per_year[0][key]
        for ya in per_year[1:]:
            acc = _add(acc, ya[key])
        total[key] = acc
    total["solo_ct"] = per_year[0]["solo_ct"]
    for ya in per_year[1:]:
        total["solo_ct"] = _add(total["solo_ct"], ya["solo_ct"])
    total["presence"] = {}
    for ya in per_year:
        for c, t in ya["presence"].items():
            total["presence"][c] = t if c not in total["presence"] else _add(total["presence"][c], t)
    lobs = [ya["lob_ct"] for ya in per_year if ya["lob_ct"] is not None]
    total["lob_ct"] = None
    for t in lobs:
        total["lob_ct"] = t if total["lob_ct"] is None else _add(total["lob_ct"], t)
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
def _is_comp_index(idx):
    cpt = idx.get_level_values("FINAL_CPT") if isinstance(idx, pd.MultiIndex) else idx
    return np.array([bool(COMPANION.match(str(c).strip().upper())) for c in cpt])


def prd_tier(t, f):
    """PRD 7.3 serving tier for one cell from labelled claim counts (TRUE t, FALSE f)."""
    n = t + f
    if n < GATE_LOW:
        return "RULES_ONLY (<100)"
    if n <= GATE_HIGH:
        return "ML 'required' only (100-500)" if t >= MINORITY_FLOOR else "below floor (<50 TRUE)"
    if t >= MINORITY_FLOOR and f >= MINORITY_FLOOR:
        return "ML both directions (>500)"
    if t >= MINORITY_FLOOR:
        return "ML 'required' only (>500, <50 FALSE)"
    if f >= MINORITY_FLOOR:
        return "ML 'not required' only (>500, <50 TRUE)"
    return "below floor (<50 TRUE)"


TIER_ORDER = ["RULES_ONLY (<100)", "below floor (<50 TRUE)", "ML 'required' only (100-500)",
              "ML 'required' only (>500, <50 FALSE)", "ML 'not required' only (>500, <50 TRUE)",
              "ML both directions (>500)"]


def plot_prd_volume_gates(prd_cells, outdir):
    """PRD 7.3 volume gates + minority-class floor, measured on the data.
    Cell = CPT + modifier + payor/plan + place of service; counts = distinct labelled claims
    (AMBIGUOUS excluded). Window = the years passed in (run --years 2026 for a single year)."""
    c = prd_cells.copy()
    t, f = c.get("TRUE", 0), c.get("FALSE", 0)
    c["n"] = t + f
    c["tier"] = [prd_tier(a, b) for a, b in zip(t, f)]
    c["p"] = t / c["n"].replace(0, np.nan)
    g = c.groupby("tier")
    pct_cells = 100 * g.size().reindex(TIER_ORDER).fillna(0) / len(c)
    pct_lab = 100 * g["n"].sum().reindex(TIER_ORDER).fillna(0) / max(c["n"].sum(), 1)
    pct_true = 100 * g["TRUE"].sum().reindex(TIER_ORDER).fillna(0) / max(t.sum(), 1)

    x = np.arange(len(TIER_ORDER))
    fig, ax = plt.subplots(figsize=(15, 6))
    for off, vals, lab, col in [(-0.27, pct_cells, "% of cells", COLOR_GRAY),
                                (0, pct_lab, "% of labelled encounters", COLOR_BLUE),
                                (0.27, pct_true, "% of PA encounters", COLOR_RED)]:
        ax.bar(x + off, vals, width=0.27, label=lab, color=col)
        for xi, v in zip(x + off, vals):
            ax.text(xi, v + 0.5, f"{v:.0f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([o.replace(" (", "\n(") for o in TIER_ORDER], fontsize=10)
    ax.set_ylabel("%"); ax.legend()
    req_tiers = ["ML 'required' only (100-500)", "ML 'required' only (>500, <50 FALSE)", "ML both directions (>500)"]
    notreq_tiers = ["ML 'not required' only (>500, <50 TRUE)", "ML both directions (>500)"]
    ml_true = float(pct_true[req_tiers].sum())
    ax.set_title(f"PRD 7.3 gates: ML may answer 'auth required' for {ml_true:.0f}% of PA claims "
                 f"(cell = CPT+modifier+plan+POS, floor {MINORITY_FLOOR})", fontsize=13)
    _save(fig, outdir, "05_prd_volume_gates")

    # cell-level ceiling: in ML tiers, how much would already meet the PRD confidence bars
    # if the model only knew the cell's own rate (>=90% for 'required', >=94% for 'not required')
    req = c[c["tier"].isin(req_tiers)]
    nreq = c[c["tier"].isin(notreq_tiers)]
    conf_req = 100 * req.loc[req["p"] >= 0.90, "TRUE"].sum() / max(t.sum(), 1)
    conf_nreq = 100 * nreq.loc[(1 - nreq["p"]) >= 0.94, "FALSE"].sum() / max(f.sum(), 1)
    c[["TRUE", "FALSE", "n", "tier", "p"]].sort_values("n", ascending=False).to_csv(f"{outdir}/05_prd_cells.csv")
    return {"cells": int(len(c)),
            "pct_cells_by_tier": pct_cells.round(1).to_dict(),
            "pct_true_claims_by_tier": pct_true.round(1).to_dict(),
            "pct_labelled_claims_by_tier": pct_lab.round(1).to_dict(),
            "ml_servable_required_pct_true": ml_true,
            "ml_servable_not_required_pct_false": float(100 * nreq["FALSE"].sum() / max(f.sum(), 1)),
            "cell_rate_meets_90pct_required_pct_true": float(conf_req),
            "cell_rate_meets_94pct_not_required_pct_false": float(conf_nreq)}


def _purity(c):
    c = c.copy()
    c["labelled"] = c.get("TRUE", 0) + c.get("FALSE", 0)
    c = c[c["labelled"] >= MIN_EVIDENCE]
    c["p_true"] = c.get("TRUE", 0) / c["labelled"]
    bins = [-0.001, 0.1, 0.5, 0.9, 1.0]
    names = ["p<0.1 (mostly no-PA)", "0.1-0.5 (mixed)", "0.5-0.9 (mostly PA)", ">=0.9 (PA)"]
    c["band"] = pd.cut(c["p_true"], bins=bins, labels=names)
    true_share = 100 * c.groupby("band", observed=False)["TRUE"].sum() / max(c["TRUE"].sum(), 1)
    lab_share = 100 * c.groupby("band", observed=False)["labelled"].sum() / max(c["labelled"].sum(), 1)
    return names, true_share, lab_share


def plot_cell_purity(prd_cells, outdir):
    """Do PRD cells alone separate PA? Distinct encounters, cells with >= MIN_EVIDENCE labelled
    encounters, weighted by PA encounters. All codes included; 'excluding flagged codes' is a side
    figure only. Purity is partly built in by the ambiguous-exclusion rule - see 14_baseline."""
    names, true_share, lab_share = _purity(prd_cells)
    _, true_x, _ = _purity(prd_cells[~_is_comp_index(prd_cells.index)])
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - 0.2, lab_share.values, width=0.4, label="% of labelled encounters", color=COLOR_BLUE)
    ax.bar(x + 0.2, true_share.values, width=0.4, label="% of PA encounters", color=COLOR_RED)
    for xi, v, vx in zip(x + 0.2, true_share.values, true_x.values):
        ax.text(xi, v + 0.5, f"{v:.0f}%\n(excl. flagged {vx:.0f}%)", ha="center", fontsize=8, color=COLOR_NAVY)
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("%"); ax.legend()
    pure = true_share[["0.5-0.9 (mostly PA)", ">=0.9 (PA)"]].sum()
    pure_x = true_x[["0.5-0.9 (mostly PA)", ">=0.9 (PA)"]].sum()
    ax.set_title(f"{pure:.0f}% of PA encounters are in cells where PA is the majority "
                 f"(PRD cells, >={MIN_EVIDENCE} encounters)", fontsize=12)
    _save(fig, outdir, "06_cell_purity")
    return {"true_in_majority_pa_cells_pct": float(pure), "excl_flagged_codes_pct": float(pure_x)}


def plot_label_by_year(per_year, outdir):
    t = pd.DataFrame({ya["year"]: ya["funnel_counts"] for ya in per_year}).T.reindex(columns=LABELS).fillna(0)
    share = 100 * t.div(t.sum(axis=1), axis=0)
    rate = 100 * t["TRUE"] / (t["TRUE"] + t["FALSE"]).replace(0, np.nan)
    pa = pd.DataFrame({ya["year"]: ya["pa_claims"] for ya in per_year}).T
    fig, (ax, axb) = plt.subplots(1, 2, figsize=(16, 5))
    share.plot(kind="bar", stacked=True, ax=ax, color=[COLOR_RED, COLOR_GREEN, COLOR_GRAY], width=0.6)
    ax.set_ylabel("% of claim lines"); ax.set_xticklabels(share.index, rotation=0)
    ax2 = ax.twinx(); ax2.plot(range(len(rate)), rate.values, color=COLOR_NAVY, marker="o", linewidth=2)
    ax2.set_ylabel("PA rate % (TRUE / (TRUE+FALSE) lines)")
    for i, v in enumerate(rate.values):
        ax2.text(i, v, f" {v:.2f}%", color=COLOR_NAVY, fontweight="bold")
    ax.set_title("Label mix and PA rate by year (lines)")
    xs = np.arange(len(pa))
    axb.bar(xs - 0.2, pa["pa_claims"], width=0.4, color=COLOR_RED, label="PA encounters (CSN with a TRUE line)")
    axb.bar(xs + 0.2, pa["pa_claims_non_companion"], width=0.4, color=COLOR_ORANGE, label="...with a TRUE line outside the flagged list")
    for xi, v in zip(xs - 0.2, pa["pa_claims"]):
        axb.text(xi, v, f"{int(v):,}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    axb.set_xticks(xs); axb.set_xticklabels(pa.index); axb.legend(fontsize=9, loc="upper left")
    enc_rate = 100 * pa["pa_claims"] / pa["labelled_claims"].replace(0, np.nan)
    axr = axb.twinx(); axr.plot(xs, enc_rate.values, color=COLOR_NAVY, marker="o", linewidth=2)
    axr.set_ylabel("% of labelled encounters with PA")
    for xi, v in zip(xs, enc_rate.values):
        axr.text(xi, v, f" {v:.1f}%", color=COLOR_NAVY, fontweight="bold", fontsize=9)
    axb.set_title("PA encounters (distinct CSN) per year - the real number of PA examples")
    _save(fig, outdir, "07_label_by_year")
    r = rate.dropna()
    return {"pa_rate_by_year_pct": r.round(3).to_dict(),
            "pa_rate_max_over_min": float(r.max() / max(r.min(), 1e-9)) if len(r) else None,
            "ambiguous_pct_by_year": share["AMBIGUOUS"].round(1).to_dict(),
            "pa_claims_by_year": pa.astype(int).to_dict(orient="index"),
            "pa_claims_non_companion_total": int(pa["pa_claims_non_companion"].sum()),
            "pa_encounters_total": int(pa["pa_claims"].sum()),
            "pa_encounters_latest_year": int(pa["pa_claims"].iloc[-1]),
            "pa_encounter_rate_by_year_pct": enc_rate.round(2).to_dict(),
            "pa_encounter_rate_max_over_min": float(enc_rate.max() / max(enc_rate.min(), 1e-9)),
            "pa_claims_non_companion_min_year": int(pa["pa_claims_non_companion"].min())}


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
    ax.set_ylabel("% of TRUE lines on flagged codes")
    ax.set_title("Share of TRUE lines on the flagged (unverified) companion list")
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
    ax.set_ylabel(f"PSI vs {latest['year']} (top 30 values + OTHER)"); ax.legend()
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
    dup_mp = sum(c.get("dup_lines_with_mod_pos", 0) for c in cl)
    mixed = sum(c["claims_mixed_label"] for c in cl)
    lab = sum(c["claims_labelled"] for c in cl)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(hist.index.astype(int).astype(str), 100 * hist.values / hist.sum(), color=COLOR_NAVY)
    ax.set_xlabel("lines per claim (20 = 20+)"); ax.set_ylabel("% of claims")
    ax.set_title(f"{n_lines/n_claims:.1f} lines/claim | repeats (claim+CPT+date) {100*dup/n_lines:.1f}% -> "
                 f"with modifier+POS {100*dup_mp/n_lines:.1f}% | mixed-label claims {100*mixed/max(lab,1):.2f}%", fontsize=11)
    _save(fig, outdir, "12_claim_structure")
    return {"lines_per_claim": n_lines / n_claims, "dup_lines_pct": 100 * dup / n_lines,
            "dup_lines_with_mod_pos_pct": 100 * dup_mp / n_lines,
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


def _avg_precision(p, t, f):
    """Average precision / PR curve from cell-level predictions p with TRUE / FALSE counts."""
    d = pd.DataFrame({"p": p, "t": t, "f": f}).groupby("p", sort=False)[["t", "f"]].sum().sort_index(ascending=False)
    tp, fp = d["t"].cumsum(), d["f"].cumsum()
    prec = tp / (tp + fp).replace(0, np.nan)
    rec = tp / max(d["t"].sum(), 1)
    ap = float((rec.diff().fillna(rec.iloc[0]) * prec.fillna(0)).sum())
    return ap, prec.values, rec.values


def _lookup_eval(train, test):
    base = train["TRUE"].sum() / max(train.values.sum(), 1)
    levels = [PRD_KEYS, ["FINAL_CPT", "PAYOR", "PLAN"], ["FINAL_CPT", "PAYOR"], ["FINAL_CPT"]]
    names = ["PRD cell", "CPT+payor+plan", "CPT+payor", "CPT"]
    te = test.reset_index()
    te["p"], te["level"] = np.nan, "base rate"
    for keys, nm in zip(levels, names):
        tr = train.groupby(level=keys).sum()
        tr = tr[(tr["TRUE"] + tr["FALSE"]) >= MIN_EVIDENCE]
        rate = (tr["TRUE"] / (tr["TRUE"] + tr["FALSE"])).rename("r").reset_index()
        te = te.merge(rate, on=keys, how="left")
        fill = te["p"].isna() & te["r"].notna()
        te.loc[fill, "p"], te.loc[fill, "level"] = te.loc[fill, "r"], nm
        te = te.drop(columns="r")
    te["p"] = te["p"].fillna(base)
    ap, prec, rec = _avg_precision(te["p"].values, te["TRUE"].values, te["FALSE"].values)
    prevalence = te["TRUE"].sum() / max((te["TRUE"] + te["FALSE"]).sum(), 1)
    lvl = 100 * te.groupby("level")["TRUE"].sum() / max(te["TRUE"].sum(), 1)
    return ap, prec, rec, prevalence, lvl.reindex(names + ["base rate"]).fillna(0)


def plot_baseline(per_year, outdir):
    """Learnability test without a model: PA rates learned from all earlier years predict the
    latest year (encounter x PRD cell). Lookup with backoff PRD cell -> CPT+payor+plan -> CPT+payor
    -> CPT -> base rate, each level used only with >= MIN_EVIDENCE labelled encounters.
    Main result uses all codes; 'excluding flagged codes' is shown alongside."""
    if len(per_year) < 2:
        logger.warning("Need 2+ years for 14_baseline - skipped")
        return {}
    train = per_year[0]["prd_cells"]
    for ya in per_year[1:-1]:
        train = _add(train, ya["prd_cells"])
    train, test = train.fillna(0), per_year[-1]["prd_cells"].fillna(0)
    ap, prec, rec, prev, lvl = _lookup_eval(train, test)
    ap_x, prec_x, rec_x, prev_x, _ = _lookup_eval(train[~_is_comp_index(train.index)],
                                                 test[~_is_comp_index(test.index)])
    def p_at(pr, rc, r):
        ok = rc >= r
        return float(np.nanmax(pr[ok])) if ok.any() else None

    fig, ax = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={"width_ratios": [3, 2]})
    ax[0].plot(rec, prec, color=COLOR_RED, linewidth=2, label=f"all codes (AP {ap:.3f})")
    ax[0].plot(rec_x, prec_x, color=COLOR_ORANGE, linewidth=1.5, linestyle=":", label=f"excl. flagged codes (AP {ap_x:.3f})")
    ax[0].axhline(prev, color=COLOR_GRAY, linestyle="--", label=f"base rate ({prev:.4f})")
    ax[0].set_xlabel("recall (share of PA encounters found)"); ax[0].set_ylabel("precision")
    ax[0].set_ylim(0, 1.02); ax[0].legend()
    ax[0].set_title(f"Train {per_year[0]['year']}-{per_year[-2]['year']} -> predict {per_year[-1]['year']}: "
                    f"AP is {ap / max(prev, 1e-9):.1f}x the base rate", fontsize=12)
    lvl.plot(kind="barh", ax=ax[1], color=COLOR_BLUE)
    ax[1].set_xlabel(f"% of {per_year[-1]['year']} PA encounters"); ax[1].set_title("Which lookup level answered")
    _save(fig, outdir, "14_baseline")
    return {"average_precision": ap, "prevalence": float(prev),
            "ap_lift_over_base": float(ap / max(prev, 1e-9)),
            "precision_at_recall_50": p_at(prec, rec, 0.5), "precision_at_recall_80": p_at(prec, rec, 0.8),
            "excl_flagged_ap_lift_over_base": float(ap_x / max(prev_x, 1e-9)),
            "pa_encounters_by_level_pct": lvl.round(1).to_dict(),
            "train_years": f"{per_year[0]['year']}-{per_year[-2]['year']}", "test_year": per_year[-1]["year"]}


def plot_cpt_solo_vs_with(solo_ct, outdir, top_n=25):
    """Evidence for 'companion' codes from the data, for EVERY CPT: PA rate when the CPT is the only
    code on the encounter vs when other codes are present. Rides along = PA almost only with others."""
    t = solo_ct.unstack("ctx").fillna(0)
    t.columns = [f"{a}_{b}" for a, b in t.columns]
    for c in ["TRUE_solo", "FALSE_solo", "TRUE_with_others", "FALSE_with_others"]:
        if c not in t:
            t[c] = 0
    t["n_solo"] = t["TRUE_solo"] + t["FALSE_solo"]
    t["n_with"] = t["TRUE_with_others"] + t["FALSE_with_others"]
    t["pa_rate_solo"] = t["TRUE_solo"] / t["n_solo"].replace(0, np.nan)
    t["pa_rate_with"] = t["TRUE_with_others"] / t["n_with"].replace(0, np.nan)
    t["pa_encounters"] = t["TRUE_solo"] + t["TRUE_with_others"]
    t["on_flagged_list"] = [bool(COMPANION.match(str(c).strip().upper())) for c in t.index]

    def pattern(r):
        if r["n_solo"] < MIN_EVIDENCE * 5 or r["pa_encounters"] < MIN_EVIDENCE:
            return "insufficient evidence"
        if r["pa_rate_with"] > 0 and r["pa_rate_solo"] < 0.2 * r["pa_rate_with"]:
            return "rides along (PA mostly with other codes)"
        return "carries PA on its own"
    t["pattern"] = t.apply(pattern, axis=1)
    t = t.sort_values("pa_encounters", ascending=False)
    t.to_csv(f"{outdir}/15_cpt_solo_vs_with.csv")

    top = t.head(top_n).iloc[::-1]
    y = np.arange(len(top))
    fig, ax = plt.subplots(figsize=(12, 0.35 * len(top) + 2))
    ax.barh(y - 0.2, 100 * top["pa_rate_solo"].fillna(0), height=0.4, color=COLOR_BLUE, label="PA rate when ONLY code on encounter")
    ax.barh(y + 0.2, 100 * top["pa_rate_with"].fillna(0), height=0.4, color=COLOR_RED, label="PA rate when other codes present")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{c}{' *' if f else ''}  ({p.split(' (')[0]})" for c, f, p in
                        zip(top.index, top["on_flagged_list"], top["pattern"])], fontsize=9)
    ax.set_xlabel("PA rate % (encounters)"); ax.legend(fontsize=9)
    ax.set_title(f"Top {top_n} CPTs by PA encounters: alone vs with other codes (* = on flagged list)", fontsize=12)
    _save(fig, outdir, "15_cpt_solo_vs_with")
    rides = t[t["pattern"].str.startswith("rides")]
    flagged = t[t["on_flagged_list"]]
    return {"cpts": int(len(t)), "cpts_ride_along": int(len(rides)),
            "pa_encounter_share_ride_along_pct": float(100 * rides["pa_encounters"].sum() / max(t["pa_encounters"].sum(), 1)),
            "flagged_codes_seen": int(len(flagged)),
            "flagged_confirmed_ride_along": int(flagged["pattern"].str.startswith("rides").sum()),
            "flagged_carry_pa": int((flagged["pattern"] == "carries PA on its own").sum()),
            "ride_along_not_on_list_top10": list(rides[~rides["on_flagged_list"]].head(10).index)}


def plot_field_presence(presence, outdir):
    """PA rate when an optional field is filled vs empty. A big gap means the field being filled is
    itself linked to PA (e.g. filled by a PA workflow) - a leakage risk to resolve before using it."""
    if not presence:
        return {}
    rows = {}
    for col, ct in presence.items():
        ct = ct.reindex(columns=["TRUE", "FALSE"]).fillna(0)
        n = ct.sum(axis=1)
        rate = 100 * ct["TRUE"] / n.replace(0, np.nan)
        rows[col] = {"fill_pct": float(100 * n.get("filled", 0) / max(n.sum(), 1)),
                     "pa_rate_filled": float(rate.get("filled", np.nan)),
                     "pa_rate_empty": float(rate.get("empty", np.nan))}
    r = pd.DataFrame(rows).T
    r["ratio"] = r["pa_rate_filled"] / r["pa_rate_empty"].replace(0, np.nan)
    x = np.arange(len(r))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - 0.2, r["pa_rate_filled"], width=0.4, color=COLOR_RED, label="PA rate when filled")
    ax.bar(x + 0.2, r["pa_rate_empty"].fillna(0), width=0.4, color=COLOR_GRAY, label="PA rate when empty")
    for xi, (_, row) in zip(x, r.iterrows()):
        ax.text(xi, max(row["pa_rate_filled"], row["pa_rate_empty"] if row["pa_rate_empty"] == row["pa_rate_empty"] else 0),
                f"filled {row['fill_pct']:.0f}%\n{row['ratio']:.1f}x" if row["ratio"] == row["ratio"] else f"filled {row['fill_pct']:.0f}%",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.3 * float(np.nanmax(r[["pa_rate_filled", "pa_rate_empty"]].values)))
    ax.set_xticks(x); ax.set_xticklabels(r.index, rotation=10); ax.set_ylabel("PA rate % (lines)"); ax.legend(loc="upper right")
    ax.set_title("Is a field being FILLED linked to PA? (big gap = leakage risk)")
    _save(fig, outdir, "16_field_presence")
    return {"by_field": r.round(3).to_dict(orient="index"),
            "max_ratio": float(r["ratio"].replace([np.inf], np.nan).max()) if r["ratio"].notna().any() else None,
            "max_ratio_field": r["ratio"].idxmax() if r["ratio"].notna().any() else None}


def plot_line_of_business(lob_ct, outdir, min_n=500):
    """PA rate by line of business (encounters). Sanity check on the label: self-pay has no payer,
    so a non-trivial PA rate there means the auth signal is catching something other than payer PA."""
    if lob_ct is None or not len(lob_ct):
        logger.warning("LINE_OF_BUSINESS not available - skipping 17_line_of_business")
        return {}
    t = lob_ct.copy()
    t["n"] = t["TRUE"] + t["FALSE"]
    t = t[t["n"] >= min_n]
    base = 100 * t["TRUE"].sum() / max(t["n"].sum(), 1)
    t["rate"] = 100 * t["TRUE"] / t["n"]
    t = t.sort_values("rate")
    colors = [COLOR_ORANGE if re.search(r"SELF|WORK|COMP|TPL|THIRD", str(i).upper()) else COLOR_BLUE for i in t.index]
    fig, ax = plt.subplots(figsize=(11, 0.35 * len(t) + 2))
    ax.barh(t.index, t["rate"], color=colors)
    ax.axvline(base, color=COLOR_NAVY, linestyle="--"); ax.text(base, -0.8, f" base {base:.2f}%", fontsize=9)
    for i, (r, n) in enumerate(zip(t["rate"], t["n"])):
        ax.text(r, i, f" {r:.2f}%  (n={int(n):,})", va="center", fontsize=8)
    ax.set_xlabel("PA rate % (encounters)")
    ax.set_title(f"PA rate by line of business (>= {min_n} encounters; orange = no health-plan PA expected)", fontsize=12)
    _save(fig, outdir, "17_line_of_business")
    sp = t[[bool(re.search(r"SELF", str(i).upper())) for i in t.index]]
    return {"base_rate_pct": float(base), "rates_pct": t["rate"].round(3).to_dict(),
            "self_pay_rate_pct": float(sp["rate"].max()) if len(sp) else None,
            "self_pay_pa_encounters": int(sp["TRUE"].sum()) if len(sp) else 0}


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
        ("Enough PA examples?", "PA encounters (distinct CSN), latest year", m["label_year"]["pa_encounters_latest_year"], _status(m["label_year"]["pa_encounters_latest_year"], 10000, 2000), ">=10k pass, >=2k warn"),
        ("Prevalence stable across years?", "max/min yearly PA rate (encounters)", round(m["label_year"]["pa_encounter_rate_max_over_min"], 2), _status(m["label_year"]["pa_encounter_rate_max_over_min"], 1.5, 3, False), "<=1.5 pass, <=3 warn; explain a shift before using old years"),
        ("Enough usable labels?", "% lines TRUE/FALSE", round(m["funnel"]["usable_pct"], 1), _status(m["funnel"]["usable_pct"], 70, 50), ">=70 pass, >=50 warn"),
        ("Label clean? (flagged codes)", "% TRUE lines on flagged list (unverified)", round(m["noise"]["companion_share_of_true_pct"], 1), "INFO", "indicative only - see 15 for data-based evidence"),
        ("Codes that ride along", "% PA encounters on CPTs that ride along (data-based)", None if not m["solo"] else round(m["solo"]["pa_encounter_share_ride_along_pct"], 1), "INFO", "candidates for the label fix; business to review"),
        ("Optional fields leak?", f"max PA-rate ratio filled/empty ({m['presence'].get('max_ratio_field', '-')})", None if not m["presence"] or m["presence"].get("max_ratio") is None else round(m["presence"]["max_ratio"], 1), _status(m["presence"].get("max_ratio"), 2, 5, False) if m["presence"] else "n/a", "<=2x pass, <=5x warn: check why filled-ness predicts PA"),
        ("Label sane for self-pay?", "PA rate % on self-pay encounters", None if not m["lob"] or m["lob"].get("self_pay_rate_pct") is None else round(m["lob"]["self_pay_rate_pct"], 2), "INFO", "should be ~0; if not, auth signal catches non-PA"),
        ("Primary vs secondary payer", "PA rate primary / secondary", f'{m["filing"].get("pa_rate_primary_pct", float("nan")):.2f}% / {m["filing"].get("pa_rate_secondary_pct", float("nan")):.2f}%', "INFO", "filter to primary if clearly higher"),
        ("Features carry signal?", "max corrected Cramér's V", round(m["assoc"]["max_cramers_v"], 3), _status(m["assoc"]["max_cramers_v"], 0.1, 0.03), ">=0.1 pass, >=0.03 warn"),
        ("Stable month to month?", "max/min monthly PA rate", m["monthly"]["monthly_rate_max_over_min"], _status(m["monthly"]["monthly_rate_max_over_min"], 2, 4, False), "<=2 pass, <=4 warn"),
        ("Stable year to year?", "max PSI payor/CPT vs latest", m["drift"].get("max_psi"), _status(m["drift"].get("max_psi"), 0.1, 0.25, False), "<=0.1 pass, <=0.25 warn"),
        ("How much PA can ML serve (PRD 7.3)?", "% PA encounters in cells where ML may say 'required'", round(m["gates"]["ml_servable_required_pct_true"], 1), _status(m["gates"]["ml_servable_required_pct_true"], 60, 30), ">=60 pass, >=30 warn"),
        ("Do cells separate PA?", "% PA encounters in majority-PA cells", round(m["purity"]["true_in_majority_pa_cells_pct"], 1), "INFO", "partly built in by the ambiguous-exclusion rule; see baseline"),
        ("Is it learnable over time?", "baseline AP / base rate (earlier years -> latest)", None if not m["baseline"] else round(m["baseline"]["ap_lift_over_base"], 1), _status(m["baseline"].get("ap_lift_over_base"), 10, 3), ">=10x pass, >=3x warn"),
        ("Features complete?", "worst feature % null", round(m["missing"]["max_null_pct"], 1), _status(m["missing"]["max_null_pct"], 5, 20, False), "<=5 pass, <=20 warn"),
        ("Missingness unrelated to label?", "max |null% TRUE - null% FALSE|", m["missing"]["max_true_false_null_gap_pp"], _status(m["missing"]["max_true_false_null_gap_pp"], 2, 10, False), "<=2pp pass, <=10pp warn"),
        ("No duplicate lines?", "% repeated lines (claim+CPT+modifier+POS+date)", round(m["claim"].get("dup_lines_with_mod_pos_pct", float("nan")), 2), _status(m["claim"].get("dup_lines_with_mod_pos_pct"), 1, 5, False), "<=1 pass, <=5 warn (units may be legitimate)"),
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


# ============================================================== remarks
def _f(v, nd=1, pct=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{v:,.{nd}f}{'%' if pct else ''}"


def write_remarks(m, sc, outdir, years):
    """REMARKS.md: one section per chart with auto-observations from the numbers, fixed caveats,
    and an empty 'Team remark' line to fill in by hand. Re-running the script overwrites it, so
    keep team remarks in a copy (e.g. REMARKS_reviewed.md)."""
    ly, g, b = m["label_year"], m["gates"], m.get("baseline") or {}
    L = [f"# Data readiness remarks - years {years[0]}-{years[-1]}", "",
         "Auto-generated by build_eda_charts.py. Observations are computed; caveats are fixed text.",
         "Copy this file before editing - a rerun overwrites it.", "",
         "Counting unit: CSN = encounter. The label is set per CSN and copied to its lines, so line counts",
         "overstate PA examples. 'Companion' codes are a hand-picked unverified list: never excluded by default.", "",
         "## Scorecard", "", "| question | value | status |", "|---|---|---|"]
    L += [f"| {r.question} | {r.value} | {r.status} |" for r in sc.itertuples()]

    def sec(title, obs, cav):
        L.extend(["", f"## {title}", "", "**Observations**"] + [f"- {o}" for o in obs] +
                 ["", "**Caveats**"] + [f"- {c}" for c in cav] + ["", "**Team remark:** ", ""])

    yr = ly["pa_claims_by_year"]
    sec("01 Funnel / 07 Label by year", [
        f"Usable label (TRUE/FALSE): {_f(m['funnel']['usable_pct'], 1, True)} of lines; TRUE = {_f(m['funnel']['true_rate_pct'], 2, True)} of usable lines.",
        "PA encounters by year: " + ", ".join(f"{y}: {v['pa_claims']:,} of {v['labelled_claims']:,} labelled" for y, v in yr.items()),
        "PA rate (encounters) by year: " + ", ".join(f"{y}: {v}%" for y, v in ly["pa_encounter_rate_by_year_pct"].items()),
        f"Max/min yearly encounter PA rate: {_f(ly['pa_encounter_rate_max_over_min'], 2)}x.",
        "AMBIGUOUS share of lines by year: " + ", ".join(f"{y}: {v}%" for y, v in ly["ambiguous_pct_by_year"].items())],
        ["A large shift between years is more likely a change in data / label construction than in payer policy - find the cause before using older years.",
         "Volume adequacy is about PA encounters per cell, not the overall rate - see 05."])
    sec("05 PRD 7.3 volume gates", [
        f"{g['cells']:,} PRD cells (CPT + modifier + payor/plan + POS).",
        f"ML may answer 'auth required' for {_f(g['ml_servable_required_pct_true'], 1, True)} of PA encounters.",
        f"ML may answer 'not required' for {_f(g['ml_servable_not_required_pct_false'], 1, True)} of non-PA encounters.",
        "PA encounters by tier: " + ", ".join(f"{k}: {v}%" for k, v in g["pct_true_claims_by_tier"].items())],
        ["Counts are distinct labelled encounters over the years in this run (PRD window to be confirmed).",
         "POS / modifier blanks are treated as their own value 'NONE'."])
    sec("06 Cell purity", [
        f"{_f(m['purity']['true_in_majority_pa_cells_pct'], 1, True)} of PA encounters sit in cells where PA is the majority "
        f"(excl. flagged codes: {_f(m['purity'].get('excl_flagged_codes_pct'), 1, True)})."],
        ["In-sample and partly built in by the ambiguous-exclusion rule; 14 is the real learnability test."])
    if b:
        sec("14 Baseline (learnability over time)", [
            f"Train {b['train_years']} -> predict {b['test_year']}: average precision {_f(b['average_precision'], 3)} vs base rate {_f(b['prevalence'], 4)} "
            f"= {_f(b['ap_lift_over_base'], 1)}x (excl. flagged codes {_f(b['excl_flagged_ap_lift_over_base'], 1)}x).",
            f"Precision at 50% recall: {_f(b['precision_at_recall_50'], 3)}; at 80% recall: {_f(b['precision_at_recall_80'], 3)}.",
            "Lookup level used: " + ", ".join(f"{k}: {v}%" for k, v in b["pa_encounters_by_level_pct"].items())],
            ["A lookup table, not a model - a floor the model must beat.",
             "If older years follow a different label regime, rerun with --years limited to the stable period."])
    if m.get("solo"):
        s15 = m["solo"]
        sec("15 CPT alone vs with other codes", [
            f"{s15['cpts_ride_along']:,} of {s15['cpts']:,} CPTs ride along (PA mostly only with other codes): "
            f"{_f(s15['pa_encounter_share_ride_along_pct'], 1, True)} of PA encounters.",
            f"Flagged list: {s15['flagged_codes_seen']} codes seen; {s15['flagged_confirmed_ride_along']} confirmed ride-along, "
            f"{s15['flagged_carry_pa']} carry PA on their own (should NOT be dropped).",
            "Ride-along codes NOT on the flagged list (top 10): " + (", ".join(s15["ride_along_not_on_list_top10"]) or "none")],
            ["Rule: ride-along if solo PA rate < 20% of with-others rate, with >= 100 solo encounters and >= 20 PA encounters.",
             "Candidate list for the label fix - for business review, not automatic removal.",
             "Full table: 15_cpt_solo_vs_with.csv."])
    if m.get("presence"):
        sec("16 Field presence (leakage check)", [
            f"{k}: filled {_f(v['fill_pct'], 1, True)}, PA rate filled {_f(v['pa_rate_filled'], 2, True)} vs empty {_f(v['pa_rate_empty'], 2, True)} "
            f"({_f(v['ratio'], 1)}x)" for k, v in m["presence"]["by_field"].items()],
            ["A high ratio means the field tends to be filled when PA is involved. Confirm with the data team when / why it is populated before using it as a feature.",
             "Any model metric quoted on 'filled-only' rows describes that subset, not the whole book."])
    if m.get("lob"):
        sec("17 Line of business", [
            f"Base PA rate {_f(m['lob']['base_rate_pct'], 2, True)} (encounters).",
            f"Self-pay PA rate: {_f(m['lob'].get('self_pay_rate_pct'), 2, True)} ({m['lob'].get('self_pay_pa_encounters', 0):,} PA encounters)."],
            ["LINE_OF_BUSINESS looked like the account financial class in earlier work; treat as context, not the insurance product.",
             "Workers' comp / third-party liability authorisations are a different process from health-plan PA."])
    sec("Other charts (02, 03, 04, 08-13)", [
        f"Top-10 payors hold {_f(m['payor']['true_share_top10_payors_pct'], 1, True)} of TRUE lines.",
        f"Strongest feature (corrected Cramér's V): {m['assoc']['best_feature']} ({_f(m['assoc']['max_cramers_v'], 3)}).",
        f"Duplicate lines incl. modifier+POS: {_f(m['claim'].get('dup_lines_with_mod_pos_pct'), 2, True)}; mixed-label encounters: {_f(m['claim'].get('mixed_label_claims_pct'), 2, True)}.",
        f"Max drift PSI vs latest year: {_f(m['drift'].get('max_psi'), 3)}."],
        ["Split train/test by encounter (CSN) and by time, never by line."])
    with open(f"{outdir}/REMARKS.md", "w") as fh:
        fh.write("\n".join(L) + "\n")
    logger.info("Saved REMARKS.md")


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
    m["gates"] = plot_prd_volume_gates(total["prd_cells"], args.outdir)
    m["purity"] = plot_cell_purity(total["prd_cells"], args.outdir)
    m["label_year"] = plot_label_by_year(per_year, args.outdir)
    m["noise"] = plot_label_noise(per_year, args.outdir)
    m["filing"] = plot_filing_order(total["filing_ct"], args.outdir)
    m["drift"] = plot_drift(per_year, args.outdir) if len(per_year) > 1 else {}
    m["missing"] = plot_missingness(per_year, total, args.outdir)
    m["claim"] = plot_claim_structure(per_year, args.outdir)
    m["coverage"] = plot_coverage(per_year, args.outdir)
    m["baseline"] = plot_baseline(per_year, args.outdir)
    m["solo"] = plot_cpt_solo_vs_with(total["solo_ct"], args.outdir)
    m["presence"] = plot_field_presence(total["presence"], args.outdir)
    m["lob"] = plot_line_of_business(total["lob_ct"], args.outdir)
    sc = build_scorecard(m, args.outdir)
    write_remarks(m, sc, args.outdir, sorted(args.years))

    with open(f"{args.outdir}/readiness_metrics.json", "w") as f:
        json.dump(m, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    logger.info("All charts saved to %s", args.outdir)


if __name__ == "__main__":
    main()
