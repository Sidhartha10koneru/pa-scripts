import pandas as pd, pyarrow.fs as pafs
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 100)

path = "cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v4/ai_prior_auth_09282026_v3_2026_std.parquet"
cols = ["CSN", "SERVICEDATE", "GROUP_LABEL", "FILING_ORDER_C", "CLAIM_SUBMISSION_DATE"]
d = pd.read_parquet(path, columns=cols, filesystem=pafs.S3FileSystem())

d["svc"] = pd.to_datetime(d.SERVICEDATE, errors="coerce")
d["sub"] = pd.to_datetime(d.CLAIM_SUBMISSION_DATE, errors="coerce")
d["label"] = d.GROUP_LABEL.astype(str).str.upper().str.strip()
d["month"] = d.svc.dt.to_period("M").astype(str)

print("rows:", f"{len(d):,}", "| service dates:", d.svc.min(), "to", d.svc.max(),
      "| unreadable dates:", f"{d.svc.isna().mean():.2%}")
print("label values:", d.label.value_counts().to_dict())

# 1) lines, encounters, label mix and PA rate per month (all claims)
m = d.groupby("month").agg(lines=("CSN", "size"), encounters=("CSN", "nunique"),
                           true=("label", lambda s: (s == "TRUE").sum()),
                           false=("label", lambda s: (s == "FALSE").sum()))
m["other_labels"] = m.lines - m.true - m.false
m["pa_line_rate"] = m.true / (m.true + m.false)
enc = d[d.label.isin(["TRUE", "FALSE"])].groupby(["month", "CSN"]).label.agg(lambda s: (s == "TRUE").any())
m["pa_enc_rate"] = enc.groupby("month").mean()
print("\n== 2026 by service month (all claims)\n", m.round(4).to_string())

# 2) same for primary claims only (what the model uses)
p = d[pd.to_numeric(d.FILING_ORDER_C, errors="coerce") == 1]
pm = p.groupby("month").agg(lines=("CSN", "size"), encounters=("CSN", "nunique"),
                            pa_line_rate=("label", lambda s: (s == "TRUE").sum() / s.isin(["TRUE", "FALSE"]).sum()))
print("\n== Primary claims only\n", pm.round(4).to_string())

# 3) how long from service to claim submission (label lag)
lag = (d.sub - d.svc).dt.days
print("\n== Days from service to claim submission, by service month (median / 90th pct)")
print(d.assign(lag=lag).groupby("month").lag.describe(percentiles=[.5, .9])[["count", "50%", "90%"]].round(0).to_string())
