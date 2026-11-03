# %% [markdown]
# # Bucket 4 (mixed cells) – checks behind each problem
# Companion to `FN_bucket4_mixed_cells_report.md`. Section numbers match the report.
# Needs in the same folder: pa_base_adapter.py, fn_root_cause.py, cell_label_check.py
# Inputs : exp_nofilter_2022/fn_mixed_cells_recall80.csv  (bucket notebook, cell 9)
#          rc_mixed/fn_causes.csv                          (python fn_root_cause.py --buckets 4 --out rc_mixed)
# Data   : std v4 files (s3 .../std_data_v4/ai_prior_auth_09282026_v3_{year}_std.parquet)

# %%
# 0 - setup
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.fs as pafs

import fn_root_cause as rc

FS = pafs.S3FileSystem()
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 60)
THERAPY = r"^(97\d{3}|9894[0-2]|92507|92526)$"

f = pd.read_csv("rc_mixed/fn_causes.csv", dtype=str)          # one row per analysed bucket-4 FN
print(f"analysed FN: {len(f):,}")


def sample_2026(n_max=2_000_000, extra=("CARC", "RARC")):
    """first ~n_max lines of the 2026 std file, with AUTH / L / SVC / primary derived (fn_root_cause.prep)"""
    d = rc.dset(2026, FS)
    parts, n = [], 0
    for b in d.to_batches(batch_size=250_000):
        t = b.to_pandas()
        t = t.loc[:, ~t.columns.duplicated()]
        t = rc.prep(t[[c for c in rc.NEED + list(extra) if c in t]])
        parts.append(t); n += len(t)
        if n > n_max:
            break
    return pd.concat(parts, ignore_index=True)


# %%
# 1 - how the 2,244 FN split into the 5 problems (label source x shared auth)
f["shared"] = f.t2_shared_with.fillna("").ne("")
code_lab = f.t1_source.isin(["PA denial code", "other remit row"])
f["problem"] = np.select(
    [code_lab,
     f.t1_source.eq("auth number") & f.shared & f.FINAL_CPT.str.match(THERAPY),
     f.t1_source.eq("auth number") & f.shared,
     f.t1_source.eq("auth number") & ~f.shared],
    ["2/3 code denial (split in 4.3)", "4 therapy (NYCE PT)", "1 claim-level auth", "5 own auth"], "other")
print(f.problem.value_counts())

# %%
# 2 - why 'what changed' does not explain bucket 4: training vs test rate of the mixed cells
c = pd.read_csv("exp_nofilter_2022/fn_mixed_cells_recall80.csv")
c["ratio_test_vs_train"] = c.test_pa_rate / c.train_pa_rate
print(c[["train_n", "train_pa_rate", "test_pa_rate", "median_score", "ratio_test_vs_train"]].describe().round(3))
print(pd.cut(c.ratio_test_vs_train, [0, .25, .5, .8, 1.25, 2, 100]).value_counts(sort=False))
print("corr(train rate, score):", c[["train_pa_rate", "median_score"]].corr().iloc[0, 1].round(2))

# %%
# 3 - label-rule review: what kind of auth values become TRUE (2026 sample)
S = sample_2026()
au = S.LINE_AUTHORIZATION_NUMBER.astype(object).fillna("").astype(str).str.upper().str.strip().str.rstrip(",. ")
L1 = S.LEVEL1_LABEL.astype(str).str.upper()
a = pd.DataFrame({"AUTH": au, "L1": L1})[au != ""]
a["kind"] = "number/code"
a.loc[a.AUTH.str.contains(r"REFERRAL|\bREF\b|PAPER"), "kind"] = "referral text"
a.loc[a.AUTH.str.contains(r"NO\s+PRIOR|NOT\s+REQ|NPN|NPAR|UNIVERSAL|NO\s*AUTH"), "kind"] = "no-auth / placeholder text"
a.loc[a.AUTH.str.fullmatch(r"(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])20\d\d"), "kind"] = "date"
a.loc[a.AUTH.str.fullmatch(r"[A-Z ]+") & a.kind.eq("number/code"), "kind"] = "other text"
print(pd.crosstab(a.kind, a.L1, margins=True))

codes = (S.CARC.astype(object).fillna("").astype(str) + " " + S.RARC.astype(object).fillna("").astype(str)).str.upper()
exact = codes.str.contains(r"(?<![A-Z0-9])(?:N54|M62)(?!\d)")
sub = codes.str.contains("N54|M62") & ~exact
print("TRUE only via N54/M62 substring (e.g. N549):",
      int((sub & S.LEVEL2_LABEL.astype(str).str.upper().eq("TRUE")).sum()))
same_dos = a.kind.eq("date") & (a.AUTH == S.loc[a.index, "SVC"].dt.strftime("%m%d%Y"))
print("auth value equal to the service date (TRUE):", int((same_dos & a.L1.eq("TRUE")).sum()))

# %%
# 4.1 - Problem 1: the auth number sits on the whole claim
p = S[S.primary]
v = p.groupby("CSN").agg(n_cpt=("FINAL_CPT", "nunique"), n_auth=("AUTH", lambda x: (x != "").sum()),
                         n_lines=("AUTH", "size"), n_vals=("AUTH", lambda x: x[x != ""].nunique()))
v = v[(v.n_cpt >= 2) & (v.n_auth > 0)]
print("visits with 2+ CPTs and an auth:", len(v))
print("auth on every line:", round((v.n_auth == v.n_lines).mean(), 3),
      "| some lines blank:", round((v.n_auth < v.n_lines).mean(), 3),
      "| 2+ different auth values:", round((v.n_vals >= 2).mean(), 3))

# %%
# 4.2 - Problem 1: which codes inherit the auth (TRUE rate alone on claim vs with other services)
s1 = S[S.L.isin(["TRUE", "FALSE"])].copy()
n = s1.groupby("CSN").FINAL_CPT.transform("nunique")
s1["visit"] = np.where(n == 1, "alone on claim", "with other services")
r = s1.assign(y=s1.L.eq("TRUE")).groupby(["FINAL_CPT", "visit"]).y.agg(["mean", "size"]).unstack()
r = r[(r["size"] >= 50).all(axis=1)]
r["carried_x"] = r[("mean", "with other services")] / r[("mean", "alone on claim")].replace(0, np.nan)
print(r.loc[r.index.isin(["99213", "99214", "99215", "96372", "36415", "G2211", "93000", "93306", "93015", "97112"])].round(3))

# example visit: injection 96372 sharing the auth with the drug Q5111
ex = f[(f.FINAL_CPT == "96372") & f.t2_shared_with.astype(str).str.contains("Q5111")]
if len(ex):
    d = rc.dset(2026, FS)
    print(rc.read(d, rc.isin(d, "CSN", [ex.CSN.iloc[0]]))[["SERVICEDATE", "FINAL_CPT", "AUTH", "L", "SPECIALTY"]].to_string())

# %%
# 4.3 - Problems 2 & 3: earlier code-denial for the same patient (analysis variable, not a model feature)
u = f[code_lab].copy()
u = u.assign(PAT_ID=u.PAT_ID.astype(str), SVC=pd.to_datetime(u.SVC, errors="coerce"))
P = pd.concat([rc.read(d, rc.isin(d, "PAT_ID", u.PAT_ID.dropna().unique()))
               for d in (rc.dset(y, FS) for y in [2025, 2026])], ignore_index=True)
den = P[(P.L == "TRUE") & P.AUTH.eq("")][["PAT_ID", "SVC", "FINAL_CPT"]].drop_duplicates()   # TRUE with no auth = code-labelled
m = u[["CSN", "PAT_ID", "SVC", "FINAL_CPT"]].merge(den, on="PAT_ID", suffixes=("", "_d"))
m = m[m.SVC_d < m.SVC - pd.Timedelta(days=45)]                                                # known before the visit
m["same_cpt"] = m.FINAL_CPT == m.FINAL_CPT_d
pr = m.groupby("CSN").agg(n_prior=("SVC_d", "size"), same_cpt=("same_cpt", "any"))
u["prior_denial"] = u.CSN.isin(pr.index)
u["prior_denial_same_cpt"] = u.CSN.map(pr.same_cpt).astype("boolean").fillna(False)
print(f"code-labelled FN {len(u)} | patients {u.PAT_ID.nunique()}")
print(pd.crosstab(u.t1_source, [u.prior_denial, u.prior_denial_same_cpt], margins=True))
print(pd.crosstab(u.PAYOR_PLAN_CLEAN, u.prior_denial).assign(n=lambda t: t.sum(1)).sort_values("n", ascending=False).head(10))

# %%
# 4.4 - Problems 2 & 3: what differs between code-denied and paid lines in the same cell
top = u.groupby(["FINAL_CPT", "PAYOR_PLAN_CLEAN"]).size().sort_values(ascending=False).head(6).reset_index()
pairs = set(zip(top.FINAL_CPT, top.PAYOR_PLAN_CLEAN))
keep = lambda t: t[pd.Series(list(zip(t.FINAL_CPT, t.PAYOR_PLAN_CLEAN)), index=t.index).isin(pairs) & t.primary]
C = pd.concat([rc.read(d, ds.field("FINAL_CPT").isin(sorted(set(top.FINAL_CPT))), keep)
               for d in (rc.dset(y, FS) for y in [2025, 2026])], ignore_index=True)
C = C[C.L.isin(["TRUE", "FALSE"])].copy()
C["grp"] = np.where((C.L == "TRUE") & C.AUTH.eq(""), "denied (code)", np.where(C.L == "FALSE", "paid (FALSE)", "auth TRUE"))
C = C[C.grp != "auth TRUE"]
npi = lambda col: C[col].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
C["provider_is_pcp"] = ~npi("PCPNPI").isin(["", "nan", "None", "<NA>"]) & (npi("SERVICEPROVIDERNPI") == npi("PCPNPI"))
for (cpt, plan), g in C.groupby(["FINAL_CPT", "PAYOR_PLAN_CLEAN"], observed=True):
    print(f"\n===== {cpt} | {plan} | denied {int((g.grp == 'denied (code)').sum())} vs paid {int((g.grp == 'paid (FALSE)').sum())}")
    print(g.groupby("grp").provider_is_pcp.mean().round(3).to_string())
    for col in ["SPECIALTY", "DEPARTMENT", "VISITTYPE", "Primary_DX"]:
        if col in g:
            t = pd.crosstab(g[col].astype(str), g.grp, normalize="columns").round(3)
            t["lift"] = (t.get("denied (code)", 0) / t.get("paid (FALSE)", np.nan)).round(1)
            print(f"-- {col}\n", t.sort_values("denied (code)", ascending=False).head(5).to_string())

# %%
# 4.5 - Problem 4: NYCE PPO physical therapy - PA starts at visit 26 (policy UMC0394)
PLAN = "EMBLEM HEALTH / GHI | NYCE PPO PLAN"
d = rc.dset(2026, FS)
T = rc.read(d, ds.field("FINAL_CPT").isin(rc.THERAPY_CODES), lambda t: t[(t.PAYOR_PLAN_CLEAN == PLAN) & t.primary])
vis = (T.groupby(["PAT_ID", "SVC"], observed=True).L.agg(lambda s: (s == "TRUE").any()).rename("pa")
       .reset_index().sort_values(["PAT_ID", "SVC"]))
vis["visit_no"] = vis.groupby("PAT_ID").cumcount() + 1          # all therapy visits this year (any label)
print("PA rate by visit number:\n", vis.groupby(pd.cut(vis.visit_no, [0, 10, 20, 25, 30, 40, 1000])).pa.agg(["mean", "size"]).round(3))
first = vis[vis.pa].groupby("PAT_ID").visit_no.min()
print("visit number of each patient's first PA:\n", first.value_counts().sort_index().head(15))
after = vis.merge(first.rename("first_pa"), on="PAT_ID")
after = after[after.visit_no > after.first_pa]
print(f"visits after a patient's first PA that are PA: {int(after.pa.sum())}/{len(after)}")
t97 = T[(T.FINAL_CPT == "97112") & T.L.isin(["TRUE", "FALSE"])]
print("monthly PA rate (97112, labelled):\n",
      (t97.L == "TRUE").groupby(t97.SVC.dt.to_period("M")).agg(["mean", "size"]).round(3))

# %%
# 4.6 - Problem 5: own auth, not shared
p5 = f[f.problem == "5 own auth"]
print(len(p5)); print(p5.cause.value_counts())
print(p5.groupby(["FINAL_CPT", "PAYOR_PLAN_CLEAN"]).size().sort_values(ascending=False).head(12))
cols = [c for c in ["FINAL_CPT", "PAYOR_PLAN_CLEAN", "cause", "t4_attr", "t4_value", "t4_rate", "t3_auth_age_days",
                    "cell_fit_rate", "cell_test_rate"] if c in p5]
print(p5[p5.cause.str.startswith("DATA")][cols].head(10).to_string())
