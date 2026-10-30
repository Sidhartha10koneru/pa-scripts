import pandas as pd, pyarrow.fs as pafs
cols = ["CSN", "FINAL_CPT", "FINAL_MODIFIERS", "SERVICEDATE", "FILING_ORDER_C", "GROUP_LABEL", "PAYOR"]
d = pd.read_parquet("cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v4/ai_prior_auth_09282026_v3_2026_std.parquet",
                    filesystem=pafs.S3FileSystem(), columns=cols)
d["L"] = d.GROUP_LABEL.astype(str).str.upper()
d = d[d.L.isin(["TRUE", "FALSE"])]
d["fo"] = pd.to_numeric(d.FILING_ORDER_C, errors="coerce")
k = ["CSN", "FINAL_CPT", "FINAL_MODIFIERS", "SERVICEDATE"]
prim = d[d.fo == 1].drop_duplicates(k).set_index(k)
sec = d[d.fo != 1].join(prim[["L", "PAYOR"]], on=k, rsuffix="_prim")

print("secondary lines:", len(sec))
print("1) same service (CSN+CPT+mod+date) also on a primary claim:", f"{sec.L_prim.notna().mean():.1%}")
m = sec[sec.L_prim.notna()]
print("2) billed to a different payor than the primary:", f"{(m.PAYOR.astype(str) != m.PAYOR_prim.astype(str)).mean():.1%}")
print("3) label agreement secondary vs primary:\n", pd.crosstab(m.L, m.L_prim, margins=True))
print("4) PA secondary lines with NO primary twin:", int(((sec.L == "TRUE") & sec.L_prim.isna()).sum()))
