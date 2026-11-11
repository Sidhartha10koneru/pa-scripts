"""
profile_visit_level_auth.py - are the VISIT_LEVEL_AUTH lines (v8 draft) referrals or real visit-level PAs?

VISIT_LEVEL_AUTH = 1: TRUE through an auth number that is shared with other CPTs of the claim, CPT rarely PA alone,
not a payor exception, BUT no other real PA line on the claim (no "owner"). Kept TRUE pending the referral decision.

Compares three groups of TRUE-through-auth-number lines (quality codes left out):
  VLA    VISIT_LEVEL_AUTH = 1                          (the 7,120 in question)
  PAX    IS_PASSENGER = 1                              (auth has an owner on the claim)
  OTHER  every other TRUE line with an auth number     (reference: ordinary PA lines)

Evidence a referral would leave (each printed side by side for the three groups):
  1. payor / plan mix         referrals are an HMO / managed-Medicaid feature (over-representation vs OTHER)
  2. auth number re-use       one referral usually covers several visits -> same number on many claims (CSNs)
  3. auth number format       shape (digits -> 9, letters -> A); text such as REF in the value
  4. claim content            which CPTs / POS; lines per claim; most common CPT sets
  5. denial codes             Acuity referral codes 165 / 243 on the line or on the same claim
  6. referral columns         any column whose name mentions REFER is listed (and profiled if present)
No raw auth numbers are printed or written; only their shapes and counts.

Run (SageMaker, same folder as build_v7_1.py):
  nohup python -u profile_visit_level_auth.py --years 2022 2023 2024 2025 2026 --out vla_profile \
     > run_vla_profile.log 2>&1 &
Outputs: vla_profile/*.csv + the log.
"""
import argparse
import os
import re
import time

import pandas as pd
import pyarrow.dataset as ds
import pyarrow.fs as pafs

from build_v7_1 import BLANK, NO_AUTH, TRUE_SET

V8 = "s3://cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v8_draft/ai_prior_auth_09282026_v3_{year}_std.parquet"
NEED = ["CSN", "FILING_ORDER_C", "FINAL_CPT", "PAYOR", "PLAN", "PLACE_OF_SERVICE_CODE", "GROUP_LABEL",
        "LINE_AUTHORIZATION_NUMBER", "IS_QUALITY_CODE", "ACUITY_AUTH_CODES", "IS_PASSENGER", "VISIT_LEVEL_AUTH"]
GROUPS = ["VLA", "PAX", "OTHER"]
REFERRAL_CODES = {"165", "243"}


def shape(v):
    """'AB12345678' -> 'A2 9-8'; keeps the pattern, never the value"""
    s = re.sub(r"[0-9]", "9", re.sub(r"[A-Z]", "A", v))
    s = re.sub(r"[^A9]", "-", s)
    return " ".join(f"{m.group(0)[0]}{len(m.group(0))}" if m.group(0)[0] != "-" else "-" for m in re.finditer(r"(.)\1*", s))


def load(year, path):
    p = path.format(year=year)
    fs, p = (pafs.S3FileSystem(), p.replace("s3://", "")) if p.startswith("s3://") else (pafs.LocalFileSystem(), p)
    d = ds.dataset(p, filesystem=fs, format="parquet")
    names = d.schema.names
    miss = [c for c in NEED if c not in names and c != "PLACE_OF_SERVICE_CODE"]
    if miss:
        raise SystemExit(f"{p}: missing {miss} - point --base-path at the v8 draft files")
    refcols = [c for c in names if "REFER" in c.upper()]
    t = d.to_table(columns=[c for c in NEED if c in names] + refcols).to_pandas()
    if "PLACE_OF_SERVICE_CODE" not in t:
        t["PLACE_OF_SERVICE_CODE"] = "?"
    return t, refcols


def prep(t, year):
    t["claim"] = (t.CSN.astype(str).str.replace(r"\.0$", "", regex=True) + "|" +
                  pd.to_numeric(t.FILING_ORDER_C, errors="coerce").fillna(1).astype(int).astype(str))
    t["cpt"] = t.FINAL_CPT.astype(str).str.strip().str.upper()
    a = t.LINE_AUTHORIZATION_NUMBER.astype(object).fillna("").astype(str).str.strip().str.upper().str.rstrip(",. ")
    t["auth"] = a.where(~a.isin(BLANK) & ~a.str.fullmatch(NO_AUTH, na=False), "")
    t["true"] = t.GROUP_LABEL.astype(str).str.strip().str.upper().isin(TRUE_SET)
    q = pd.to_numeric(t.IS_QUALITY_CODE, errors="coerce").fillna(0).eq(1)
    vla = pd.to_numeric(t.VISIT_LEVEL_AUTH, errors="coerce").fillna(0).eq(1)
    pax = pd.to_numeric(t.IS_PASSENGER, errors="coerce").fillna(0).eq(1)
    t["grp"] = ""
    base = t.true & t.auth.ne("") & ~q
    t.loc[base, "grp"] = "OTHER"
    t.loc[base & pax, "grp"] = "PAX"
    t.loc[base & vla, "grp"] = "VLA"
    t["payor"] = t.PAYOR.astype(str).str.upper().str.replace(r"\bAS OF\b.*$", "", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
    t["plan"] = t.payor + " | " + t.PLAN.astype(str).str.upper().str.replace(r"\bAS OF\b.*$", "", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
    codes = t.ACUITY_AUTH_CODES.astype(object).fillna("").astype(str)
    t["ref_code_line"] = codes.str.split(",").map(lambda l: bool(REFERRAL_CODES & set(l)))
    t["year"] = year
    return t


def share_table(g, col, top=25):
    """counts and share of each group by col, plus VLA over-representation vs REF"""
    c = g.groupby([col, "grp"], observed=True).size().unstack(fill_value=0).reindex(columns=GROUPS, fill_value=0)
    s = c / c.sum()
    out = c.add_suffix("_lines").join(s.add_suffix("_share"))
    out["VLA_vs_OTHER"] = (out.VLA_share / out.OTHER_share.replace(0, float("nan"))).round(2)
    return out.sort_values("VLA_lines", ascending=False).head(top)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, required=True)
    ap.add_argument("--base-path", default=V8)
    ap.add_argument("--out", default="vla_profile")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20); pd.set_option("display.max_colwidth", 60)

    keep, claims, refcols_all = [], [], set()
    for y in a.years:
        t0 = time.time()
        t, refcols = load(y, a.base_path)
        refcols_all |= set(refcols)
        t = prep(t, y)
        # claim-level facts for claims that hold a VLA or PAX line
        cl = t[t.grp.isin(["VLA", "PAX"])].groupby("claim").grp.agg(lambda s: "VLA" if "VLA" in set(s) else "PAX")
        ct = t[t.claim.isin(cl.index)]
        cf = ct.groupby("claim").agg(lines=("cpt", "size"), n_cpt=("cpt", "nunique"),
                                     cpts=("cpt", lambda s: " + ".join(sorted(set(s))[:6])),
                                     ref_code_claim=("ref_code_line", "max")).join(cl.rename("grp"))
        cf["payor"] = ct.groupby("claim").payor.first()
        claims.append(cf.assign(year=y))
        g = t[t.grp != ""][["year", "claim", "CSN", "cpt", "auth", "grp", "payor", "plan", "PLACE_OF_SERVICE_CODE",
                            "ref_code_line"] + refcols]
        keep.append(g)
        print(f"[{y}] lines {len(t):,} | TRUE-by-auth lines: " +
              " ".join(f"{k} {int((g.grp == k).sum()):,}" for k in GROUPS) + f" | {time.time() - t0:.0f}s")
        del t, ct
    g = pd.concat(keep, ignore_index=True)
    cf = pd.concat(claims)
    g["CSN"] = g.CSN.astype(str).str.replace(r"\.0$", "", regex=True)

    print("\n== 0. size")
    print(g.groupby(["year", "grp"]).size().unstack(fill_value=0).reindex(columns=GROUPS).to_string())

    print("\n== 1. payor / plan mix (VLA_vs_OTHER > 1 = more common among VLA lines than among ordinary PA lines)")
    p = share_table(g, "payor"); p.to_csv(f"{a.out}/by_payor.csv"); print(p.to_string())
    pl = share_table(g, "plan", top=40); pl.to_csv(f"{a.out}/by_plan.csv")
    print(f"  top-10 payors hold {p.VLA_share.head(10).sum():.1%} of VLA lines (by_plan.csv has plans)")

    print("\n== 2. auth number re-use: distinct claims (CSN) per auth value, by the group of the line")
    per_auth = g.groupby("auth").CSN.nunique().rename("csn_per_auth")
    g = g.join(per_auth, on="auth")
    bins = pd.cut(g.csn_per_auth, [0, 1, 2, 5, 10, 50, 10**9], labels=["1", "2", "3-5", "6-10", "11-50", ">50"])
    r = pd.crosstab(bins, g.grp, normalize="columns").reindex(columns=GROUPS).round(3)
    r.loc["median"] = g.groupby("grp").csn_per_auth.median().reindex(GROUPS)
    r.to_csv(f"{a.out}/auth_reuse.csv"); print(r.to_string())
    print("  (re-use is counted within the TRUE-by-auth lines only; one auth on many visits points to a referral or a"
          " standing authorisation, not a one-off procedure PA)")

    print("\n== 3. auth number format (shape only)")
    g["shape"] = g.auth.map(shape)
    g["has_ref_text"] = g.auth.str.contains(r"REF", na=False)
    g["len"] = g.auth.str.len()
    sh = share_table(g, "shape", top=20); sh.to_csv(f"{a.out}/auth_shapes.csv"); print(sh.to_string())
    print(g.groupby("grp").agg(lines=("auth", "size"), auth_values=("auth", "nunique"), median_len=("len", "median"),
                               ref_text=("has_ref_text", "mean")).reindex(GROUPS).round(3).to_string())

    print("\n== 4. what is on the lines and claims")
    c = share_table(g, "cpt", top=20); c.to_csv(f"{a.out}/by_cpt.csv"); print(c.to_string())
    pos = share_table(g, "PLACE_OF_SERVICE_CODE", top=10); pos.to_csv(f"{a.out}/by_pos.csv"); print(pos.to_string())
    print(cf.groupby("grp").agg(claims=("lines", "size"), median_lines=("lines", "median"),
                                median_cpts=("n_cpt", "median")).to_string())
    combos = cf[cf.grp == "VLA"].cpts.value_counts().head(20)
    combos.to_csv(f"{a.out}/vla_claim_cpt_sets.csv"); print("  most common CPT sets on VLA claims:\n" + combos.to_string())

    print("\n== 5. referral denial codes 165 / 243 (Acuity list)")
    print(pd.DataFrame({"on_line": g.groupby("grp").ref_code_line.mean(),
                        "on_claim": cf.groupby("grp").ref_code_claim.mean()}).reindex(GROUPS).round(4).to_string())

    print("\n== 6. columns whose name mentions REFER:", sorted(refcols_all) or "none")
    for col in sorted(refcols_all):
        v = g[col].astype(object).fillna("").astype(str).str.strip().ne("")
        print(f"  {col}: filled on " + ", ".join(f"{k} {v[g.grp == k].mean():.1%}" for k in GROUPS))
    g.groupby(["grp", "payor", "cpt"]).size().rename("lines").reset_index().sort_values("lines", ascending=False) \
        .to_csv(f"{a.out}/grp_payor_cpt.csv", index=False)
    print(f"\noutputs -> {a.out}/")


if __name__ == "__main__":
    main()
