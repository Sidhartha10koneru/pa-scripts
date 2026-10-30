import pandas as pd, pyarrow.fs as pafs
import pa_base_adapter as pb

fs = pafs.S3FileSystem()
cols = [pb.CSN, pb.LABEL, pb.DATE, pb.FILING, "FINAL_CPT", "PAYOR", "PLAN"]
d = pd.read_parquet(pb.BASE_PATH.format(year=2026).replace("s3://", ""), filesystem=fs, columns=cols)

d["SVC_DATE"] = pd.to_datetime(d[pb.DATE], errors="coerce")
d = d[d[pb.FILING].astype(str).str.strip().isin(["1", "1.0"])]                         # primary claims
d = d[(d.SVC_DATE >= "2026-05-01") & (d.SVC_DATE < "2026-08-01")]                       # test window
d["L"] = d[pb.LABEL].astype(str).str.upper()

# 1) line level
print("TEST lines:", len(d))
print(d.L.value_counts().to_string())
print((d.L.value_counts(normalize=True) * 100).round(2).to_string())

# 2) visit level (a visit is TRUE if any line is TRUE)
v = d.groupby(pb.CSN).L.agg(lambda x: "TRUE" if (x == "TRUE").any() else ("FALSE" if (x == "FALSE").any() else "AMBIGUOUS"))
print("\nTEST visits:", len(v))
print(v.value_counts().to_string())

# 3) same AMBIGUOUS check as 2025: groups with PA vs without
K = ["FINAL_CPT", "PAYOR", "PLAN"]
d["t"] = d.L.eq("TRUE")
d["has_pa"] = d.groupby(K, observed=True).t.transform("max")
t = d.groupby("has_pa").L.value_counts().unstack(fill_value=0)
t["n"] = t.sum(1)
t["amb_share"] = t.get("AMBIGUOUS", 0) / t.n
t["pa_rate_labelled"] = t.get("TRUE", 0) / (t.get("TRUE", 0) + t.get("FALSE", 0))
t["pa_rate_if_amb_false"] = t.get("TRUE", 0) / t.n
print("\n", t.round(3).to_string())

# 4) 99214 in test
print("\n99214:", d[d.FINAL_CPT.astype(str) == "99214"].L.value_counts().to_string())


import pandas as pd, pyarrow.fs as pafs
import pa_base_adapter as pb

fs = pafs.S3FileSystem()
d = pd.read_parquet(pb.BASE_PATH.format(year=2026).replace("s3://", ""), filesystem=fs,
                    columns=[pb.CSN, pb.LABEL, pb.DATE, pb.FILING, "FINAL_CPT", "FINAL_MODIFIERS"])
d["SVC_DATE"] = pd.to_datetime(d[pb.DATE], errors="coerce")
d["T"] = d[pb.LABEL].astype(str).str.upper().eq("TRUE")

def show(step, x):
    print(f"{step:<45} TRUE lines {x.T.sum():>9,} | TRUE visits {x.loc[x.T, pb.CSN].nunique():>9,}")

show("1. whole 2026 file", d)
x = d[(d.SVC_DATE >= "2026-05-01") & (d.SVC_DATE < "2026-08-01")]
show("2. May-Jul only", x)
x = x[x[pb.FILING].astype(str).str.strip().isin(["1", "1.0"])]
show("3. + primary claims only", x)
x = x.drop_duplicates([pb.CSN, "FINAL_CPT", "FINAL_MODIFIERS", "SVC_DATE"])
show("4. + repeats removed (= our test set)", x)
print("\nAug rows in file:", (d.SVC_DATE >= "2026-08-01").sum(), "| date range:", d.SVC_DATE.min(), "->", d.SVC_DATE.max())
