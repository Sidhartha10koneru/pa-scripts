"""
label_audit.py - audit of the PA labels (GROUP_LABEL) for one year.

Two parts, run separately or together:

  PART A  (std file, fast)  AMBIGUOUS audit
    - per CPT: TRUE / FALSE / AMBIGUOUS counts, PA rate "as labelled" (AMBIGUOUS dropped)
      and "if AMBIGUOUS = FALSE" (lower bound), share of lines dropped
    - the same per CPT + payor + plan (where the inflation actually happens)
    - which remit codes (CARC) the AMBIGUOUS lines carry, in groups WITH PA vs WITHOUT PA
    - a known-PA check (MRI, PET, Botox...): does their PA rate stay high if AMBIGUOUS = FALSE?
    - extract of all AMBIGUOUS lines (parquet) for the CPTs you choose

  PART B  (raw remit file)  own vs passenger TRUE
    For every claim line with PA evidence, says WHY it is PA:
      own_denial      the line's own remit rows carry a PA denial code (197, 15, 39 ...)
      own_auth        it has an auth number used by no other CPT in the same visit
      shared_anchor   it shares an auth number with other CPTs in the visit and is the "main" service
      shared_passenger it shares an auth number and is NOT the main service -> PA only because of another CPT
    The main service (anchor) in a shared auth = the CPT with the highest clinical rank
    (procedure / imaging / drug / rehab > infusion admin > E/M visit > lab > contrast & supplies > quality F codes);
    ties are broken by how often that CPT has its OWN evidence across the year.
    E.g. MRI 70553 + contrast A9579 + quality code 3074F on one auth -> 70553 anchor, A9579 and 3074F passengers.
    (Rank first, because an MRI almost always shares its auth with its contrast, so its "own" rate is low.)

Run (from the folder with your S3 access):
  python label_audit.py columns --year 2025                       # list the columns of both files first
  python label_audit.py ambiguous --year 2025 --out audit_2025
  python label_audit.py passenger --year 2025 --out audit_2025 --auth-col <AUTH COLUMN NAME>
  python label_audit.py all --year 2025 --out audit_2025 --auth-col <AUTH COLUMN NAME> --focus-cpt 99214
  add --sample 0.2 to use 20% of visits if memory is tight (whole visits are kept)
"""
import argparse
import os
import re

import numpy as np
import pandas as pd

STD = "cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v4/ai_prior_auth_09282026_v3_{year}_std.parquet"
RAW = "cli-ml-data-prod/datalake/ai-prior-auth/09282026/v3/ai_prior_auth_09282026_v3_{year}.parquet"

PA_CARC = {"15", "39", "197", "198", "210", "284", "296", "302"}       # same list as the labelling script
PA_RARC = {"M62", "N54"}
KNOWN_PA = ["70553", "70551", "72148", "72141", "73721", "74177", "78815", "78452", "64615", "99214"]
NO_AUTH = r"^(|NAN|NONE|NULL|N/?A|0+|-+|\.|NO|NO ?AUTH.*|NOT ?REQ.*|NO ?PA.*|NONE ?REQ.*|NAR|UNKNOWN|PENDING)$"


# ----------------------------------------------------------------- helpers
def fs():
    import pyarrow.fs as pafs
    return pafs.S3FileSystem()


def read(path, cols, sample=1.0, csn="CSN"):
    import pyarrow.parquet as pq
    names = set(pq.read_schema(path, filesystem=fs()).names) if not os.path.exists(path) else set(pq.read_schema(path).names)
    use = [c for c in dict.fromkeys(cols) if c in names]
    missing = [c for c in cols if c not in names]
    if missing:
        print(f"  (columns not in {os.path.basename(path)}: {missing})")
    d = pd.read_parquet(path, columns=use, filesystem=None if os.path.exists(path) else fs())
    if sample < 1:                                               # keep whole visits
        keep = (pd.util.hash_pandas_object(d[csn].astype(str), index=False) % 1000) < sample * 1000
        d = d[keep.values]
    return d


def codes(s):
    """'CO-197, MA15' -> {'197', 'MA15'}   (exact codes, group prefix CO/PR/OA/PI/CR removed)"""
    s = s.astype(str).str.upper().str.replace(r"\b(CO|PR|OA|PI|CR)[-\s]*(?=\d)", "", regex=True)
    return s.str.findall(r"[A-Z]*\d+[A-Z]?")


def family(c):
    c = str(c).upper()
    if re.fullmatch(r"\d{4}F", c): return "quality code (F)"
    if re.fullmatch(r"G\d{4}", c): return "HCPCS G"
    if re.fullmatch(r"99[2-4]\d\d", c): return "E/M office/hospital visit"
    if re.fullmatch(r"7\d{4}", c): return "radiology/imaging"
    if re.fullmatch(r"8\d{4}", c): return "lab/pathology"
    if re.fullmatch(r"97\d{3}", c): return "PT/OT rehab"
    if re.fullmatch(r"96\d{3}", c): return "infusion/injection admin"
    if re.fullmatch(r"[A]\d{4}", c): return "HCPCS A (supplies/contrast)"
    if re.fullmatch(r"Q\d{4}", c): return "HCPCS Q"
    if re.fullmatch(r"J\d{4}", c): return "HCPCS J drug"
    if re.fullmatch(r"[1-6]\d{4}", c): return "surgery/procedure"
    if re.fullmatch(r"9\d{4}", c): return "medicine other"
    return "other"


# which CPT is the "main" service when several share one auth: clinical rank first, then own-evidence rate
ANCHOR_RANK = {"surgery/procedure": 1, "radiology/imaging": 1, "HCPCS J drug": 1, "PT/OT rehab": 1,
               "infusion/injection admin": 2, "medicine other": 2, "HCPCS G": 2, "E/M office/hospital visit": 3,
               "lab/pathology": 4, "other": 4, "HCPCS A (supplies/contrast)": 5, "HCPCS Q": 5, "quality code (F)": 6}


def save(df, out, name, show=25, msg=""):
    df.to_csv(f"{out}/{name}.csv", index=isinstance(df.index, pd.MultiIndex) or df.index.name is not None)
    print(f"\n== {name}  {msg}\n{df.head(show).round(4).to_string()}")


# ----------------------------------------------------------------- PART A
def part_ambiguous(a):
    print(f"\n######## PART A - AMBIGUOUS audit, std file {a.year}")
    cols = ["CSN", a.cpt_col, "FINAL_MODIFIERS", "PAYOR", "PLAN", "GROUP_LABEL", "CARC", "RARC",
            "FILING_ORDER_C", "SERVICEDATE", "GROUP_GUID"]
    d = read(STD.format(year=a.year), cols, a.sample)
    d["L"] = d.GROUP_LABEL.astype(str).str.strip().str.upper().map(
        {"TRUE": "TRUE", "1": "TRUE", "FALSE": "FALSE", "0": "FALSE"}).fillna("AMBIGUOUS")
    d["CPT"] = d[a.cpt_col].astype(str)
    print(f"  {len(d):,} lines  |  " + "  ".join(f"{k} {v:,} ({v/len(d):.1%})" for k, v in d.L.value_counts().items()))

    def rates(g):
        t = g.L.value_counts().reindex(["TRUE", "FALSE", "AMBIGUOUS"], fill_value=0)
        n = t.sum()
        return pd.Series({"lines": n, "TRUE": t.TRUE, "FALSE": t.FALSE, "AMBIGUOUS": t.AMBIGUOUS,
                          "amb_share": t.AMBIGUOUS / n,
                          "pa_rate_as_labelled": t.TRUE / max(t.TRUE + t.FALSE, 1),
                          "pa_rate_if_amb_false": t.TRUE / n})

    # 1) groups with vs without PA (the core finding)
    K = ["CPT", "PAYOR", "PLAN"]
    d["grp_has_pa"] = d.groupby(K, observed=True).L.transform(lambda s: (s == "TRUE").any())
    g = d.groupby("grp_has_pa").apply(rates)
    g["inflation_x"] = g.pa_rate_as_labelled / g.pa_rate_if_amb_false.replace(0, np.nan)
    save(g, a.out, "A1_groups_with_vs_without_pa", msg="(CPT+payor+plan groups)")

    # 2) per CPT
    c = d.groupby("CPT").apply(rates)
    c["inflation_x"] = c.pa_rate_as_labelled / c.pa_rate_if_amb_false.replace(0, np.nan)
    c["family"] = c.index.map(family)
    c = c.sort_values("TRUE", ascending=False)
    save(c, a.out, "A2_per_cpt", msg="(sorted by TRUE lines)")

    # 3) per CPT + payor + plan, for the CPTs you care about (or the top 30 by TRUE)
    focus = a.focus_cpt or list(c.index[:30])
    p = d[d.CPT.isin(focus)].groupby(K, observed=True).apply(rates)
    p = p[p.TRUE > 0].sort_values("TRUE", ascending=False)
    save(p, a.out, "A3_per_cpt_payor_plan", show=30, msg=f"(CPTs: {focus[:10]}{'...' if len(focus) > 10 else ''})")

    # 4) which codes the AMBIGUOUS lines carry (std keeps the first remit row's CARC/RARC)
    amb = d[d.L == "AMBIGUOUS"]
    for col in ("CARC", "RARC"):
        if col not in amb:
            continue
        x = amb[[col, "grp_has_pa"]].assign(code=amb[col].astype(str).str.strip().str.upper())
        t = x.groupby(["code", "grp_has_pa"]).size().unstack(fill_value=0)
        t.columns = [f"lines_in_groups_{'with' if k else 'without'}_PA" for k in t.columns]
        t["total"] = t.sum(1)
        t = t.sort_values("total", ascending=False)
        t["cum_share"] = t.total.cumsum() / t.total.sum()
        save(t, a.out, f"A4_ambiguous_codes_{col}", msg="(codes on AMBIGUOUS lines)")

    # 5) known-PA safety check for the "paid normally -> FALSE" rule
    k = c.loc[c.index.intersection(KNOWN_PA), ["lines", "TRUE", "FALSE", "AMBIGUOUS", "pa_rate_as_labelled",
                                                "pa_rate_if_amb_false"]]
    save(k, a.out, "A5_known_pa_check",
         msg="(MRI/PET/Botox should stay high if AMBIGUOUS=FALSE; if they collapse, auths are missing on paid lines)")

    # 6) extract AMBIGUOUS lines
    ex = amb if not a.extract_cpts else amb[amb.CPT.isin(a.extract_cpts)]
    ex = ex.drop(columns=["L"])
    ex.to_parquet(f"{a.out}/A6_ambiguous_lines.parquet", index=False)
    print(f"\n== A6 extract: {len(ex):,} AMBIGUOUS lines -> {a.out}/A6_ambiguous_lines.parquet"
          + (f" (CPTs {a.extract_cpts})" if a.extract_cpts else " (all CPTs)"))


# ----------------------------------------------------------------- PART B
def part_passenger(a):
    print(f"\n######## PART B - own vs passenger TRUE, raw remit file {a.year}")
    if not a.auth_col:
        raise SystemExit("PART B needs --auth-col (the auth number column in the raw file; run 'columns' to see it)")
    line_key = ["CSN", a.raw_cpt_col] + [c for c in a.line_keys if c]
    d = read(RAW.format(year=a.year), line_key + [a.auth_col, "CARC", "RARC"], a.sample)
    line_key = [c for c in line_key if c in d]
    print(f"  {len(d):,} remit rows; a claim line = {line_key}")
    d["CPT"] = d[a.raw_cpt_col].astype(str)

    # evidence per remit row
    au = d[a.auth_col].astype(str).str.strip().str.upper()
    d["auth"] = au.where(~au.str.fullmatch(a.no_auth_regex), None)
    carc = codes(d["CARC"]) if "CARC" in d else pd.Series([[]] * len(d), index=d.index)
    rarc = codes(d["RARC"]) if "RARC" in d else pd.Series([[]] * len(d), index=d.index)
    d["denial"] = [bool(PA_CARC & set(x)) or bool(PA_RARC & set(y)) or bool(PA_RARC & set(x))
                   for x, y in zip(carc, rarc)]

    # one row per claim line
    ln = d.groupby(line_key, dropna=False).agg(
        CPT=("CPT", "first"), denial=("denial", "any"),
        auth=("auth", lambda s: s.dropna().iloc[0] if s.notna().any() else None)).reset_index()
    ln["has_auth"] = ln.auth.notna()
    # how many DIFFERENT CPTs in the same visit carry the same auth number
    au_n = ln[ln.has_auth].groupby(["CSN", "auth"]).CPT.nunique().rename("cpts_on_auth")
    ln = ln.join(au_n, on=["CSN", "auth"])
    ln["auth_shared"] = ln.cpts_on_auth.fillna(0) > 1

    # own evidence
    ln["evidence"] = np.select(
        [ln.denial, ln.has_auth & ~ln.auth_shared, ln.has_auth & ln.auth_shared],
        ["own_denial", "own_auth", "shared_auth"], "none")
    own_rate = ln.assign(own=ln.evidence.isin(["own_denial", "own_auth"])).groupby("CPT").own.mean()
    ln["cpt_own_rate"] = ln.CPT.map(own_rate)
    # anchor score: clinical rank (procedure/imaging/drug > E/M > lab > contrast/supplies > quality codes),
    # ties broken by how often the CPT has its own evidence across the year
    ln["anchor_score"] = (10 - ln.CPT.map(family).map(ANCHOR_RANK).fillna(4)) + ln.cpt_own_rate
    # anchor vs passenger inside each shared auth
    s = ln.evidence == "shared_auth"
    top = ln[s].groupby(["CSN", "auth"]).anchor_score.transform("max")
    ln.loc[s, "evidence"] = np.where(ln.loc[s, "anchor_score"] >= top, "shared_anchor", "shared_passenger")
    # which CPT is the anchor each passenger rides on
    anc = ln[ln.evidence == "shared_anchor"].sort_values("anchor_score", ascending=False) \
        .drop_duplicates(["CSN", "auth"]).set_index(["CSN", "auth"]).CPT.rename("anchor_cpt")
    ln = ln.join(anc, on=["CSN", "auth"])
    ln["family"] = ln.CPT.map(family)

    pa = ln[ln.evidence != "none"]
    print(f"  claim lines {len(ln):,}; with PA evidence {len(pa):,}")
    save(pa.evidence.value_counts().to_frame("lines").assign(share=lambda x: x.lines / x.lines.sum()),
         a.out, "B1_evidence_overall")

    # per CPT: how its TRUE lines are made up
    t = pd.crosstab(pa.CPT, pa.evidence)
    for col in ["own_denial", "own_auth", "shared_anchor", "shared_passenger"]:
        t[col] = t.get(col, 0)
    t["pa_lines"] = t.sum(1)
    t["own_share"] = (t.own_denial + t.own_auth) / t.pa_lines
    t["passenger_share"] = t.shared_passenger / t.pa_lines
    t["family"] = t.index.map(family)
    t["top_anchor"] = pa[pa.evidence == "shared_passenger"].groupby("CPT").anchor_cpt.agg(
        lambda x: x.value_counts().head(3).to_dict())
    t = t[t.pa_lines >= a.min_lines].sort_values("pa_lines", ascending=False)
    save(t, a.out, "B2_per_cpt", msg=f"(CPTs with >= {a.min_lines} PA lines)")
    save(t.sort_values("passenger_share", ascending=False), a.out, "B3_most_passenger_cpts",
         msg="(PA mostly because of ANOTHER cpt on the same auth)")

    # per family
    f = pd.crosstab(pa.family, pa.evidence)
    f["pa_lines"] = f.sum(1)
    f["passenger_share"] = f.get("shared_passenger", 0) / f.pa_lines
    save(f.sort_values("pa_lines", ascending=False), a.out, "B4_per_family")

    # passenger -> anchor pairs
    pr = pa[pa.evidence == "shared_passenger"].groupby(["CPT", "anchor_cpt"]).size().rename("lines") \
        .sort_values(ascending=False).to_frame()
    pr["passenger_family"] = [family(i[0]) for i in pr.index]
    pr["anchor_family"] = [family(i[1]) for i in pr.index]
    save(pr, a.out, "B5_passenger_anchor_pairs", show=30)

    # line-level detail for chosen CPTs
    for cpt in a.focus_cpt or []:
        x = ln[ln.CPT == cpt]
        x.to_csv(f"{a.out}/B6_lines_{cpt}.csv", index=False)
        print(f"\n== B6 {cpt}: {len(x):,} lines -> {a.out}/B6_lines_{cpt}.csv\n"
              f"{x.evidence.value_counts().to_string()}")
        if (x.evidence == "shared_passenger").any():
            print(f"   rides on: {x[x.evidence == 'shared_passenger'].anchor_cpt.value_counts().head(10).to_dict()}")


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["columns", "ambiguous", "passenger", "all"])
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--out", default=None)
    ap.add_argument("--sample", type=float, default=1.0, help="share of visits kept (whole visits)")
    ap.add_argument("--cpt-col", default="FINAL_CPT", help="CPT column in the std file")
    ap.add_argument("--raw-cpt-col", default="FINAL_CPT", help="CPT column in the raw remit file")
    ap.add_argument("--auth-col", default=None, help="auth number column in the raw remit file")
    ap.add_argument("--line-keys", nargs="*", default=["FINAL_MODIFIERS", "SERVICEDATE", "FILING_ORDER_C"],
                    help="extra columns that identify one claim line in the raw file")
    ap.add_argument("--no-auth-regex", default=NO_AUTH, help="auth values treated as 'no auth'")
    ap.add_argument("--focus-cpt", nargs="*", default=[], help="CPTs for per-plan table and line detail, e.g. 99214")
    ap.add_argument("--extract-cpts", nargs="*", default=[], help="limit the AMBIGUOUS extract to these CPTs")
    ap.add_argument("--min-lines", type=int, default=50)
    a = ap.parse_args()

    if a.mode == "columns":
        import pyarrow.parquet as pq
        for name, p in (("std", STD), ("raw", RAW)):
            sch = pq.read_schema(p.format(year=a.year), filesystem=fs())
            print(f"\n{name}: {p.format(year=a.year)}\n  " + "\n  ".join(sch.names))
        return
    a.out = a.out or f"label_audit_{a.year}"
    os.makedirs(a.out, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    if a.mode in ("ambiguous", "all"):
        part_ambiguous(a)
    if a.mode in ("passenger", "all"):
        part_passenger(a)
    print(f"\nAll outputs in {a.out}/")


if __name__ == "__main__":
    main()
  python label_audit.py columns --year 2025
nohup python -u label_audit.py all --year 2025 --auth-col <AUTH_COLUMN> --focus-cpt 99214 70553 \
  --out label_audit_2025 > run_label_audit_2025.log 2>&1 &

