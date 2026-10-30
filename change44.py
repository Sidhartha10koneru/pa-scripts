import pandas as pd, pyarrow.fs as pafs
import pa_base_adapter as pb

d = pd.read_parquet(pb.BASE_PATH.format(year=2026).replace("s3://", ""), filesystem=pafs.S3FileSystem(),
                    columns=[pb.CSN, pb.LABEL, pb.DATE, pb.FILING, "FINAL_CPT", "FINAL_MODIFIERS"])
d["SVC_DATE"] = pd.to_datetime(d[pb.DATE], errors="coerce")
d["L"] = d[pb.LABEL].astype(str).str.upper()
d["is_pa"] = d.L.eq("TRUE")

def row(step, x):
    return {"step": step, "lines": len(x), "PA lines": int(x.is_pa.sum()),
            "visits": x[pb.CSN].nunique(), "PA visits": x.loc[x.is_pa, pb.CSN].nunique()}

rows = []
x = d[(d.SVC_DATE >= "2026-05-01") & (d.SVC_DATE < "2026-08-01")]
x = x[x[pb.FILING].astype(str).str.strip().isin(["1", "1.0"])]
rows.append(row("1. May-Jul 2026, primary claims", x))
x = x[x.L != "AMBIGUOUS"]
rows.append(row("2. AMBIGUOUS lines dropped", x))
x = x.drop_duplicates([pb.CSN, "FINAL_CPT", "FINAL_MODIFIERS", "SVC_DATE"])
rows.append(row("3. repeat lines removed (= model test set)", x))
f = pd.DataFrame(rows).set_index("step")
f["PA lines per PA visit"] = (f["PA lines"] / f["PA visits"]).round(2)
print(f.to_string())

# proof that lines -> visits is grouping, not loss: how many PA lines each PA visit has
k = d.loc[(d.SVC_DATE >= "2026-05-01") & (d.SVC_DATE < "2026-08-01") & d.is_pa].groupby(pb.CSN).size()
print("\nPA lines per PA visit:\n", k.clip(upper=6).value_counts().sort_index().rename({6: "6+"}).to_string())

# example: one PA visit with several PA lines
ex = k[k.between(3, 5)].index[0]
print("\nExample visit:\n", d.loc[d[pb.CSN] == ex, ["FINAL_CPT", "FINAL_MODIFIERS", "SVC_DATE", "L"]].to_string())
