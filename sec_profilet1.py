"""
sec_profile.py - primary vs secondary claims across the WHOLE dataset (all years, no model).  Counts and rates only.

How lines link:  a visit (CSN) can be billed to several payers; each payer gets its own claim, marked by FILING_ORDER_C
(1 = primary, 2 = secondary, 3+ = later). The same procedure on two claims of one visit = same CSN + same CPT, so
primary and secondary are linked at SERVICE level (CSN + CPT). CSN alone links the claims of a visit, not the lines.

  T1_lines_by_filing     lines, labelled lines, PA rate per filing order (line level)
  T2_service_kinds       services (CSN + CPT) with primary only / primary + secondary / secondary only, PA rate
  T3_csn_kinds           visits (CSN): primary claim only / + secondary claim; does the secondary claim carry the same
                         CPTs as the primary (all / some / extra CPTs only on secondary)
  T4_label_agreement     primary + secondary services: primary label x secondary label, and auth number same / differs
  T5_payer_roles         each payer | plan as PRIMARY vs as SECONDARY payer: lines and services, PA / not PA / PA rate
                         (labelled only; top 30 by secondary services, all payers in T5_payer_roles_full.csv)
  T6_payer_pairs         primary payer -> secondary payer: services, PA rate (top 30)
  T7_plan_cpt_roles      plan + CPT: PA rate as primary vs as secondary (top 30 by secondary volume; full list in *_full.csv)
  T7b_high_PA_plan_cpt   same, only combinations that are >= 30% PA as primary (>= 50 services) - the ones the model scores high
  T8_by_year             T2 per year
PA label default LABEL_V6_PAX; AMBIGUOUS / missing labels left out of rates. Auth values never printed.

  nohup python -u sec_profile.py --out sec_profile_imag --family IMAG > run_sec_profile.log 2>&1 &    # radiology
  nohup python -u sec_profile.py --out sec_profile_all > run_sec_profile_all.log 2>&1 &               # all families
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.fs as pafs

from build_pa_history import family

V8 = "s3://cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v8_draft/ai_prior_auth_09282026_v3_{year}_std.parquet"
NO_AUTH = r"^(|NONE|NA|N/A|NAR|NULL|NAN|0+|NO AUTH.*|NOT REQ.*|N\.?A\.?|AUTH NOT REQ.*|NO PA.*|RFRL REQD)$"


def key(s):
    return s.astype(object).where(s.notna(), None).map(lambda v: "" if v is None else str(v)) \
        .str.strip().str.replace(r"^(\d+)\.0$", r"\1", regex=True)


def fast_map(s, fn):
    codes, uniq = pd.factorize(s.astype(object).fillna(""))
    u = pd.Series(uniq, dtype=object).map(fn)
    return pd.Series(u.values[codes], index=s.index)


def clean_name(v):
    import re
    v = re.sub(r"\bAS OF\b.*$", "", str(v).upper())
    return re.sub(r"\s+", " ", v).strip()


def rate(pa, n):
    return (pa / n.replace(0, np.nan)).round(4)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-path", default=V8)
    ap.add_argument("--years", nargs="+", type=int, default=[2022, 2023, 2024, 2025, 2026])
    ap.add_argument("--family", nargs="*", default=[])
    ap.add_argument("--label", default="LABEL_V6_PAX")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 60)

    T1, SV, CS, AG, PR, PP, PC = [], [], [], [], [], [], []
    for y in a.years:
        t0 = time.time()
        p = a.base_path.format(year=y)
        fs, pp = (pafs.S3FileSystem(), p.replace("s3://", "")) if p.startswith("s3://") else (pafs.LocalFileSystem(), os.path.abspath(p))
        dset = ds.dataset(pp, filesystem=fs, format="parquet")
        cols = [c for c in ["CSN", "FINAL_CPT", "FILING_ORDER_C", "PAYOR", "PLAN", a.label, "LINE_AUTHORIZATION_NUMBER"] if c in dset.schema.names]
        x = dset.to_table(columns=cols).to_pandas()
        x["k_cpt"] = fast_map(x.FINAL_CPT, lambda v: clean_name(v).replace(".0", "") if str(v).endswith(".0") else clean_name(v))
        if a.family:
            x = x[fast_map(x.k_cpt, family).isin(a.family)].copy()
        x["k_csn"] = key(x.CSN)
        fo = fast_map(x.FILING_ORDER_C, lambda v: str(v).replace(".0", "").strip())
        x["role"] = np.where(fo.eq("1"), "P", np.where(fo.isin(["2", "3", "4", "5"]), "S", "O"))
        x["payer"] = fast_map(x.PAYOR, clean_name) + " | " + fast_map(x.PLAN, clean_name)
        lv = fast_map(x[a.label], lambda v: str(v).strip().upper())
        x["lab"] = np.where(lv.isin(["TRUE", "1", "1.0"]), 1.0, np.where(lv.isin(["FALSE", "0", "0.0"]), 0.0, np.nan))
        if "LINE_AUTHORIZATION_NUMBER" in x:
            au = fast_map(x.LINE_AUTHORIZATION_NUMBER, lambda v: str(v).strip().upper().rstrip(",. ") if v is not None else "")
            x["auth"] = au.where(~au.str.fullmatch(NO_AUTH) & au.str.contains(r"\d"), np.nan)
        else:
            x["auth"] = np.nan

        # T1 lines
        l1 = x.groupby(x.FILING_ORDER_C.astype(str).str.replace(".0", "", regex=False)).agg(lines=("lab", "size"), labelled=("lab", "count"), PA=("lab", "sum"))
        T1.append(l1)

        # services: primary side / secondary side
        g = ["k_csn", "k_cpt"]
        P = x[x.role == "P"].groupby(g).agg(p_lab=("lab", "max"), p_payer=("payer", "first"), p_auth=("auth", "first"))
        Q = x[x.role == "S"].groupby(g).agg(s_lab=("lab", "max"), s_payer=("payer", "first"), s_auth=("auth", "first"))
        S = P.join(Q, how="outer")
        S["has_p"], S["has_s"] = S.p_payer.notna(), S.s_payer.notna()
        S["kind"] = np.select([S.has_p & S.has_s, S.has_p], ["primary + secondary", "primary only"], "secondary only")
        S["lab"] = S[["p_lab", "s_lab"]].max(axis=1)
        sv = S.groupby("kind").agg(services=("lab", "size"), labelled=("lab", "count"), PA=("lab", "sum"))
        SV.append(sv.assign(year=y))

        # T3 visits
        c = S.reset_index().groupby("k_csn").agg(n_both=("kind", lambda k: (k == "primary + secondary").sum()),
                                                 n_ponly=("kind", lambda k: (k == "primary only").sum()),
                                                 n_sonly=("kind", lambda k: (k == "secondary only").sum()),
                                                 labelled=("lab", "count"), PA=("lab", "max"))
        c["visit"] = np.select([(c.n_both == 0) & (c.n_sonly == 0), (c.n_both == 0) & (c.n_ponly == 0),
                                (c.n_ponly == 0) & (c.n_sonly == 0), c.n_sonly > 0],
                               ["primary claim only", "secondary claim only", "secondary carries ALL primary CPTs",
                                "secondary has CPTs not on primary"], "secondary carries SOME primary CPTs")
        CS.append(c.groupby("visit").agg(visits=("n_both", "size"), labelled=("labelled", lambda z: int((z > 0).sum())),
                                         PA=("PA", lambda z: int((z == 1).sum()))))

        # T4 agreement
        B = S[(S.kind == "primary + secondary") & S.p_lab.notna() & S.s_lab.notna()].copy()
        B["pattern"] = np.where(B.p_lab == 1, "primary PA", "primary not PA") + " / " + np.where(B.s_lab == 1, "secondary PA", "secondary not PA")
        B["auth"] = np.select([B.p_auth.isna() & B.s_auth.isna(), B.p_auth.eq(B.s_auth), B.p_auth.isna(), B.s_auth.isna()],
                              ["no auth on either", "SAME auth on both", "auth on secondary only", "auth on primary only"], "different auth numbers")
        AG.append(B.groupby(["pattern", "auth"]).size())

        # T5 payer roles, T6 pairs, T7 plan + CPT roles (labelled services only)
        Sp = S[S.has_p & S.lab.notna()]
        Ss = S[S.has_s & S.lab.notna()]
        xl = x[x.lab.notna()]
        PR.append(pd.concat([Sp.groupby("p_payer").lab.agg(["size", "sum"]).add_prefix("prim_svc_"),
                             Ss.groupby("s_payer").lab.agg(["size", "sum"]).add_prefix("sec_svc_"),
                             xl[xl.role == "P"].groupby("payer").lab.agg(["size", "sum"]).add_prefix("prim_line_"),
                             xl[xl.role == "S"].groupby("payer").lab.agg(["size", "sum"]).add_prefix("sec_line_")], axis=1))
        Bl = S[(S.kind == "primary + secondary") & S.lab.notna()]
        PP.append(Bl.groupby(["p_payer", "s_payer"]).lab.agg(["size", "sum"]))
        Sp2 = Sp.reset_index().assign(pc=lambda z: z.p_payer + " || " + z.k_cpt).groupby("pc").lab.agg(["size", "sum"]).add_prefix("prim_")
        Ss2 = Ss.reset_index().assign(pc=lambda z: z.s_payer + " || " + z.k_cpt).groupby("pc").lab.agg(["size", "sum"]).add_prefix("sec_")
        PC.append(pd.concat([Sp2, Ss2], axis=1))
        print(f"[{y}] lines {len(x):,} | services {len(S):,} | primary + secondary {int((S.kind == 'primary + secondary').sum()):,} | {time.time() - t0:.0f}s")
        del x, S, P, Q, B

    out = {}
    t1 = pd.concat(T1).groupby(level=0).sum()
    t1["share"] = (t1.lines / t1.lines.sum()).round(4)
    t1["PA_rate"] = rate(t1.PA, t1.labelled)
    out["T1_lines_by_filing"] = t1.sort_values("lines", ascending=False)
    sv = pd.concat(SV)
    t2 = sv.groupby(level=0)[["services", "labelled", "PA"]].sum()
    t2["share"] = (t2.services / t2.services.sum()).round(4)
    t2["PA_rate"] = rate(t2.PA, t2.labelled)
    out["T2_service_kinds"] = t2
    t3 = pd.concat(CS).groupby(level=0).sum()
    t3["share"] = (t3.visits / t3.visits.sum()).round(4)
    t3["PA_rate"] = rate(t3.PA, t3.labelled)          # visit = PA if any of its services is PA
    out["T3_csn_kinds"] = t3.sort_values("visits", ascending=False)
    out["T4_label_agreement"] = pd.concat(AG).groupby(level=[0, 1]).sum().unstack(fill_value=0)
    pr = pd.concat(PR).groupby(level=0).sum()
    t5 = pd.DataFrame(index=pr.index)
    for role, nm in (("prim", "primary"), ("sec", "secondary")):
        for lvl, ln in (("line", "lines"), ("svc", "services")):
            n, pa = pr[f"{role}_{lvl}_size"], pr[f"{role}_{lvl}_sum"]
            t5[f"{nm}_{ln}"] = n.astype(int)
            t5[f"{nm}_{ln}_PA"] = pa.astype(int)
            t5[f"{nm}_{ln}_not_PA"] = (n - pa).astype(int)
            t5[f"{nm}_{ln}_PA_rate"] = rate(pa, n)
    t5.to_csv(f"{a.out}/T5_payer_roles_full.csv")
    out["T5_payer_roles"] = t5.sort_values("secondary_services", ascending=False).head(30)
    pp = pd.concat(PP).groupby(level=[0, 1]).sum()
    pp["PA_rate"] = rate(pp["sum"], pp["size"])
    out["T6_payer_pairs"] = pp.rename(columns={"size": "services", "sum": "PA"}).sort_values("services", ascending=False).head(30)
    pc = pd.concat(PC).groupby(level=0).sum()
    pc["PA_rate_as_primary"] = rate(pc.prim_sum, pc.prim_size)
    pc["PA_rate_as_secondary"] = rate(pc.sec_sum, pc.sec_size)
    full = pc.rename(columns={"prim_size": "services_as_primary", "sec_size": "services_as_secondary"}) \
        [["services_as_primary", "PA_rate_as_primary", "services_as_secondary", "PA_rate_as_secondary"]] \
        .sort_values("services_as_secondary", ascending=False)
    full.to_csv(f"{a.out}/T7_plan_cpt_roles_full.csv")
    out["T7_plan_cpt_roles"] = full.head(30)
    # where it matters: plan + CPT combinations that are usually PA when the plan is primary
    hi = full[(full.services_as_primary >= 50) & (full.PA_rate_as_primary >= 0.3) & (full.services_as_secondary >= 20)].copy()
    hi["drop_points"] = ((hi.PA_rate_as_primary - hi.PA_rate_as_secondary) * 100).round(1)
    out["T7b_high_PA_plan_cpt"] = hi.head(30)
    t8 = sv.reset_index().pivot_table(index="year", columns="kind", values="services", aggfunc="sum")
    pa8 = sv.reset_index().assign(r=lambda z: z.PA / z.labelled).pivot_table(index="year", columns="kind", values="r").round(4)
    out["T8_by_year"] = t8.join(pa8.add_suffix(" PA rate"))

    for k, v in out.items():
        v.to_csv(f"{a.out}/{k}.csv")
        print(f"\n== {k}\n{v.to_string()}")
    print(f"\noutputs -> {a.out}/ (counts and rates only)")
    os._exit(0)


if __name__ == "__main__":
    main()
