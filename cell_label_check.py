"""
cell_label_check.py - how are the labels of one cell made, and do they agree with the raw remit data?

Default cell: 99214 + no modifier + POS 11 + "EMBLEM HEALTH/HIP | EMBLEM HEALTH/HIP", service years 2025-2026.

Part 1 (std v4 file)
  - lines, PA, PA rate; patient groups: never PA / PA patient before-at PA / after a PA visit (prior_pa)
  - label source per line (LEVEL1_LABEL, LEVEL2_LABEL, N_TRUE, N_FALSE, N_AMBIGUOUS) by group
  - CARC on the std file (first remit row) by group
  - paid amount by group (if a paid / allowed column exists)
Part 2 (raw v3 remit file, all remit rows of the same lines)
  - all CARC / RARC codes and auth values per line, not only the first row
  - per group: share of lines with a PA denial code, an auth number, only paid-normally codes, blank codes
  - first-row CARC (std) vs all codes (raw)
  - a simple re-derived label from raw vs GROUP_LABEL (agreement table)

Run:
  python cell_label_check.py --out cell_check_emblem99214
  python cell_label_check.py --cpt 99213 --plan "EMBLEM HEALTH/HIP | EMBLEM HEALTH/HIP" --out cell_check_emblem99213
  python cell_label_check.py --schema-only          # print the columns of both files and stop
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.fs as pafs
import pyarrow.parquet as pq

import pa_base_adapter as pb

STD = "cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v4/ai_prior_auth_09282026_v3_{year}_std.parquet"
RAW = "cli-ml-data-prod/datalake/ai-prior-auth/09282026/v3/ai_prior_auth_09282026_v3_{year}.parquet"
PA_CODES = {"15", "39", "197", "198", "210", "284", "296", "302", "M62", "N54"}
PAID_NORMAL = {"45", "1", "2", "3", "23", "223", "253", "MA01", "MA15", "MA18"}
BLANK = {"", "NAN", "NONE", "NULL", "<NA>"}
NO_AUTH = r"(NO\s*(REF|REFERRAL|AUTH|PA)\b.*|NOT\s*REQ.*|NAR|N/?A|NONE|0+)"


def fs_():
    return pafs.S3FileSystem()


def schema(path, fs):
    return pq.read_schema(path, filesystem=fs).names


def pick(names, *cands):
    """first column name that exists (case-insensitive)"""
    up = {n.upper(): n for n in names}
    for c in cands:
        if c.upper() in up:
            return up[c.upper()]
    return None


def toks(s):
    """'CO-197, MA15' -> {'197','MA15'}"""
    s = str(s).upper()
    s = re.sub(r"\b(CO|PR|OA|PI|CR)[-\s]*(?=\d)", "", s)
    return set(re.findall(r"[A-Z]*\d+[A-Z]?", s))


def show(df, name, out, msg=""):
    df.to_csv(f"{out}/{name}.csv")
    print(f"\n== {name} {msg}\n{df.to_string()}")


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--cpt", default="99214")
    a.add_argument("--plan", default="EMBLEM HEALTH/HIP | EMBLEM HEALTH/HIP", help="PAYOR_PLAN_CLEAN value")
    a.add_argument("--pos", default="11")
    a.add_argument("--modifier", default="", help="'' = no modifier")
    a.add_argument("--years", nargs="+", type=int, default=[2025, 2026])
    a.add_argument("--out", default="cell_check")
    a.add_argument("--schema-only", action="store_true")
    o = a.parse_args()
    os.makedirs(o.out, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    fs = fs_()

    std_names = schema(STD.format(year=o.years[0]), fs)
    raw_names = schema(RAW.format(year=o.years[0]), fs)
    if o.schema_only:
        print("STD columns:\n ", "\n  ".join(std_names))
        print("\nRAW columns:\n ", "\n  ".join(raw_names))
        return

    # ------------------------------------------------------------- Part 1: std file
    parts = []
    for y in o.years:
        d = ds.dataset(STD.format(year=y), filesystem=fs, format="parquet")
        parts.append(d.to_table(filter=ds.field("FINAL_CPT") == o.cpt).to_pandas())
    d = pd.concat(parts, ignore_index=True)
    d = d.loc[:, ~d.columns.duplicated()]
    d = pb.clean_names(d)
    mod = d.FINAL_MODIFIERS.astype(object).fillna("").astype(str).str.strip()
    mod = mod.where(~mod.str.upper().isin(BLANK), "")
    x = d[(d.PAYOR_PLAN_CLEAN == o.plan) & (d.PLACE_OF_SERVICE_CODE.astype(str).str.strip() == o.pos)
          & (mod == o.modifier)].copy()
    x["L"] = x.GROUP_LABEL.astype(str).str.upper()
    print(f"cell {o.cpt} | mod '{o.modifier}' | POS {o.pos} | {o.plan} | years {o.years}")
    print("all labels:", x.L.value_counts().to_dict())
    x = x[x.L.isin(["TRUE", "FALSE"])].copy()
    x["y"] = (x.L == "TRUE").astype(int)
    print(f"labelled lines {len(x):,} | PA {x.y.sum():,} | rate {x.y.mean():.2%}")
    if len(x) == 0:
        raise SystemExit("no labelled lines in this cell - check --plan / --pos / --modifier spelling")

    x["SVC"] = pd.to_datetime(x.SERVICEDATE, errors="coerce")
    x = x.sort_values(["PAT_ID", "SVC"]) if "PAT_ID" in x else x.sort_values("SVC")
    if "PAT_ID" in x:
        x["prior_pa"] = x.groupby("PAT_ID").y.transform(lambda s: s.shift().cummax()).fillna(0)
        x["pat_has_pa"] = x.groupby("PAT_ID").y.transform("max") == 1
        x["group"] = np.select([~x.pat_has_pa, x.prior_pa == 1],
                               ["1 patient never PA", "3 after a PA visit"], "2 PA patient before/at PA")
    else:
        x["group"] = "all"
    show(x.groupby(["group", "y"]).size().unstack(fill_value=0), "1_groups", o.out, "(lines by group and label)")

    src = [c for c in ["LEVEL1_RAW_LABEL", "LEVEL1_LABEL", "LEVEL2_LABEL"] if c in x]
    for lab, nm in ((0, "FALSE"), (1, "TRUE")):
        s = x[x.y == lab]
        if src and len(s):
            show(pd.crosstab([s[c].astype(str) for c in src], s.group, margins=True), f"2_source_{nm}", o.out,
                 f"(how {nm} lines were labelled; std = first remit row)")
    for c in ["N_TRUE", "N_FALSE", "N_AMBIGUOUS"]:
        if c in x:
            show(pd.crosstab(x[c], x.L), f"3_{c}", o.out, "(remit rows per line with this label)")

    f = x[x.y == 0]
    show(pd.crosstab(f.CARC.astype(object).fillna("").astype(str).str.upper().replace(["NAN", ""], "(blank)"), f.group, margins=True)
         .sort_values("All", ascending=False).head(25), "4_std_carc_false", o.out, "(first-row CARC on FALSE lines)")

    paid = pick(x.columns, "PAID_AMOUNT", "PAIDAMOUNT", "PAYMENT_AMOUNT", "LINE_PAID_AMOUNT", "PAID_AMT",
                "ALLOWED_AMOUNT", "ALLOWEDAMOUNT")
    if paid:
        v = pd.to_numeric(x[paid], errors="coerce")
        show(x.assign(paid_gt0=v > 0, paid_missing=v.isna()).groupby(["group", "L"])[["paid_gt0", "paid_missing"]]
             .mean().round(3), "5_paid", o.out, f"(share paid > 0 using {paid})")
    else:
        print("\n(no paid / allowed amount column found in std; candidates:",
              [c for c in x.columns if any(k in c.upper() for k in ["PAID", "ALLOW", "PAY", "AMOUNT", "AMT"])], ")")

    # ------------------------------------------------------------- Part 2: raw remit file
    csn_r = pick(raw_names, "CSN", "CSN_ID")
    cpt_r = pick(raw_names, "FINAL_CPT", "CPT", "CPT_CODE", "PROC_CODE")
    mod_r = pick(raw_names, "FINAL_MODIFIERS", "MODIFIERS", "MODIFIER")
    dt_r = pick(raw_names, "SERVICEDATE", "SERVICE_DATE", "DOS")
    fo_r = pick(raw_names, "FILING_ORDER_C", "FILING_ORDER")
    au_r = pick(raw_names, "LINE_AUTHORIZATION_NUMBER", "AUTHORIZATION_NUMBER", "AUTH_NUMBER", "PRIOR_AUTH_NUMBER")
    carc_r, rarc_r = pick(raw_names, "CARC"), pick(raw_names, "RARC")
    paid_r = pick(raw_names, "PAID_AMOUNT", "PAIDAMOUNT", "PAYMENT_AMOUNT", "LINE_PAID_AMOUNT", "PAID_AMT")
    print(f"\nRAW columns used: CSN={csn_r} CPT={cpt_r} MOD={mod_r} DATE={dt_r} FILING={fo_r} AUTH={au_r} "
          f"CARC={carc_r} RARC={rarc_r} PAID={paid_r}")
    if not (csn_r and cpt_r and carc_r):
        raise SystemExit("raw file is missing CSN / CPT / CARC under the expected names - run --schema-only")

    csns = x.CSN.dropna().unique().tolist()
    cols = [c for c in [csn_r, cpt_r, mod_r, dt_r, fo_r, au_r, carc_r, rarc_r, paid_r] if c]
    rparts = []
    for y in o.years:
        rd = ds.dataset(RAW.format(year=y), filesystem=fs, format="parquet")
        t = rd.to_table(columns=cols, filter=ds.field(cpt_r) == o.cpt).to_pandas()
        rparts.append(t[t[csn_r].astype(str).isin(set(map(str, csns)))])
    r = pd.concat(rparts, ignore_index=True)
    print(f"raw remit rows for these lines: {len(r):,}")

    # one line key on both sides
    def key(df, c_csn, c_mod, c_dt, c_fo):
        m = df[c_mod].astype(object).fillna("").astype(str).str.strip().str.upper().replace(list(BLANK), "") if c_mod else ""
        dt = pd.to_datetime(df[c_dt], errors="coerce").dt.strftime("%Y-%m-%d") if c_dt else ""
        fo = pd.to_numeric(df[c_fo], errors="coerce").fillna(-1).astype(int).astype(str) if c_fo else ""
        return df[c_csn].astype(str) + "|" + m + "|" + dt + "|" + fo

    r["k"] = key(r, csn_r, mod_r, dt_r, fo_r)
    x["k"] = key(x, "CSN", "FINAL_MODIFIERS", "SERVICEDATE", "FILING_ORDER_C" if fo_r else None)

    r["codes"] = [toks(c) | (toks(rr) if rarc_r else set()) for c, rr in zip(r[carc_r].astype(object).fillna(""), r[rarc_r].astype(object).fillna("") if rarc_r else [""] * len(r))]
    au = r[au_r].astype(object).fillna("").astype(str).str.strip().str.upper() if au_r else pd.Series("", index=r.index)
    r["auth_real"] = ~au.isin(BLANK) & ~au.str.fullmatch(NO_AUTH, na=False)
    r["auth_noauth_text"] = au.str.fullmatch(NO_AUTH, na=False) & ~au.isin(BLANK)
    r["row_blank_codes"] = r.codes.map(len) == 0
    g = r.groupby("k").agg(
        remit_rows=("codes", "size"),
        all_codes=("codes", lambda s: ",".join(sorted(set().union(*s)))),
        any_pa_code=("codes", lambda s: any(PA_CODES & c for c in s)),
        only_paid_normal=("codes", lambda s: all(c <= PAID_NORMAL for c in s) and any(len(c) for c in s)),
        any_blank_row=("row_blank_codes", "any"),
        any_auth=("auth_real", "any"),
        any_noauth_text=("auth_noauth_text", "any"),
        auths=(au_r, lambda s: ",".join(sorted(set(s.astype(str)) - {"", "nan", "None"}))[:80]) if au_r else ("codes", "size"))
    if paid_r:
        g["raw_paid_gt0"] = r.groupby("k")[paid_r].apply(lambda s: (pd.to_numeric(s, errors="coerce") > 0).any())

    m = x.merge(g, left_on="k", right_index=True, how="left")
    print(f"std lines matched to raw: {m.remit_rows.notna().mean():.1%}")
    m["raw_label"] = np.select([m.any_auth | m.any_pa_code, m.any_noauth_text, m.any_blank_row, m.only_paid_normal],
                               ["TRUE (auth or PA code)", "FALSE (explicit no-auth text)", "FALSE (a row with no codes)",
                                "paid-normally codes only"], "other / ambiguous")
    m.loc[m.remit_rows.isna(), "raw_label"] = "not found in raw"

    show(pd.crosstab(m.raw_label, [m.L], margins=True), "6_raw_vs_label", o.out,
         "(raw re-derived evidence vs GROUP_LABEL)")
    show(pd.crosstab(m.raw_label, m.group, margins=True), "7_raw_evidence_by_group", o.out,
         "(evidence behind each group, all remit rows)")
    fm = m[m.y == 0]
    cnt = fm[["all_codes", "group"]].dropna().assign(code=lambda t: t.all_codes.str.split(",")).explode("code").reset_index(drop=True)
    cnt["code"] = cnt.code.replace("", "(none)")
    show(pd.crosstab(cnt.code, cnt.group, margins=True).sort_values("All", ascending=False).head(30),
         "8_raw_codes_false", o.out, "(ALL remit codes on FALSE lines, by group)")
    if paid_r:
        show(m.groupby(["group", "L"]).raw_paid_gt0.mean().round(3).to_frame(), "9_raw_paid", o.out,
             "(share of lines with any remit row paid > 0)")

    keep = [c for c in ["GROUP_GUID", "CSN", "PAT_ID", "SERVICEDATE", "FILING_ORDER_C", "SPECIALTY", "Primary_DX",
                        "LINE_AUTHORIZATION_NUMBER", "CARC", "LEVEL1_LABEL", "LEVEL2_LABEL", "N_TRUE", "N_FALSE",
                        "L", "group", "prior_pa", "remit_rows", "all_codes", "auths", "any_auth", "any_pa_code",
                        "only_paid_normal", "any_blank_row", "raw_label"] + (["raw_paid_gt0"] if paid_r else []) if c in m]
    m[keep].to_csv(f"{o.out}/lines_detail.csv", index=False)
    print(f"\nline-level detail -> {o.out}/lines_detail.csv")
    if "PAT_ID" in m:
        ex = m[m.group == "3 after a PA visit"].PAT_ID.unique()[:5]
        print("\n== example patients (PA then follow-ups)\n",
              m[m.PAT_ID.isin(ex)][[c for c in ["PAT_ID", "SERVICEDATE", "L", "SPECIALTY", "Primary_DX",
                                                 "auths", "all_codes", "raw_label"] if c in m]].to_string(index=False))


if __name__ == "__main__":
    main()
