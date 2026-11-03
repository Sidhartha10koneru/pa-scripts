import pyarrow.fs as pafs, fn_root_cause as rc, numpy as np, pandas as pd
fs = pafs.S3FileSystem()
top = u.groupby(["FINAL_CPT", "PAYOR_PLAN_CLEAN"]).size().sort_values(ascending=False).head(6).reset_index()
cpts, pairs = set(top.FINAL_CPT), set(zip(top.FINAL_CPT, top.PAYOR_PLAN_CLEAN))
keep = lambda t: t[pd.Series(list(zip(t.FINAL_CPT, t.PAYOR_PLAN_CLEAN)), index=t.index).isin(pairs) & t.primary]
C = pd.concat([rc.read(d, rc.ds.field("FINAL_CPT").isin(sorted(cpts)), keep) for d in (rc.dset(y, fs) for y in [2025, 2026])])
C = C[C.L.isin(["TRUE", "FALSE"])].copy()
C["grp"] = np.where((C.L == "TRUE") & C.AUTH.eq(""), "denied (code)", np.where(C.L == "FALSE", "paid (FALSE)", "auth TRUE"))
C = C[C.grp != "auth TRUE"]
npi = lambda c: C[c].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
C["pcp_known"] = ~npi("PCPNPI").isin(["", "nan", "None", "<NA>"])
C["provider_is_pcp"] = C.pcp_known & (npi("SERVICEPROVIDERNPI") == npi("PCPNPI"))

for (cpt, plan), g in C.groupby(["FINAL_CPT", "PAYOR_PLAN_CLEAN"], observed=True):
    print(f"\n===== {cpt} | {plan} | denied {int((g.grp=='denied (code)').sum())} vs paid {int((g.grp=='paid (FALSE)').sum())}")
    print(g.groupby("grp")[["pcp_known", "provider_is_pcp"]].mean().round(3).to_string())
    for col in ["SPECIALTY", "DEPARTMENT", "VISITTYPE", "Primary_DX"]:
        if col in g:
            t = pd.crosstab(g[col].astype(str), g.grp, normalize="columns").round(3)
            t["lift"] = (t.get("denied (code)", 0) / t.get("paid (FALSE)", np.nan)).round(1)
            print(f"-- {col} (share of each group; lift = denied/paid)")
            print(t.sort_values("denied (code)", ascending=False).head(5).to_string())
