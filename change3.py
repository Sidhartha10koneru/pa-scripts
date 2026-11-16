python - <<'EOF'
import pandas as pd, pyarrow.fs as pafs
fs = pafs.S3FileSystem()
P = "cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v8_draft/ai_prior_auth_09282026_v3_{y}_std.parquet"
T = {"TRUE", "1", "1.0"}
parts = []
for y in [2022, 2023, 2024, 2025, 2026]:
    d = pd.read_parquet(P.format(y=y), columns=["FINAL_CPT","PAYOR","LABEL_V6","LABEL_V6_PAX"], filesystem=fs)
    d["cpt"] = d.FINAL_CPT.astype(str).str.strip().str.upper()
    d = d[~d.cpt.str.fullmatch(r"\d{4}F")]                                  # quality codes excluded, as in training
    d["payor"] = d.PAYOR.astype(str).str.upper().str.replace(r"\bAS OF\b.*$", "", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
    d["before"] = d.LABEL_V6.astype(str).str.strip().str.upper().isin(T)
    d["after"]  = d.LABEL_V6_PAX.astype(str).str.strip().str.upper().isin(T)
    parts.append(d.groupby(["cpt","payor"]).agg(lines=("before","size"), pa_before=("before","sum"), pa_after=("after","sum")))
    print(y, "done")
c = pd.concat(parts).groupby(level=[0,1]).sum()
for name, g in [("CPT", c.groupby(level=0).sum()), ("CPT + payer", c)]:
    g = g[g.pa_before > 0].assign(lost=lambda x: 1 - x.pa_after / x.pa_before)
    zero = g[g.pa_after == 0].sort_values("pa_before", ascending=False)
    print(f"\n== {name}: {len(g):,} with PA before | lost ALL PA: {len(zero):,} "
          f"({int(zero.pa_before.sum()):,} PA lines) | lost >=90%: {int((g.lost >= 0.9).sum()):,}")
    print(zero.head(25).to_string())
    zero.to_csv(f"pax_zero_pa_{name.replace(' + ','_').lower()}.csv")
EOF
