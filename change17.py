s = pd.read_parquet(path_v4_2026, columns=["FINAL_CPT","PAYOR","PLAN","GROUP_LABEL","N_TRUE","N_FALSE","N_AMBIGUOUS"],
                    filesystem=pafs.S3FileSystem())
s["lab"] = s.GROUP_LABEL.astype(str)
s["flipped"] = (s.lab == "FALSE") & (s.N_FALSE == 0) & (s.N_TRUE == 0)          # FALSE only via override
s["orig_false"] = (s.lab == "FALSE") & ~s.flipped                              # FALSE before override
cell = s.groupby(["FINAL_CPT","PAYOR","PLAN"], observed=True).agg(orig_false=("orig_false","sum"),
                                                                  flipped=("flipped","sum"))
print("lines flipped:", f"{s.flipped.sum():,}")
print("…of which in cells with NO original FALSE:", f"{cell.loc[cell.orig_false == 0, 'flipped'].sum():,}")
print(cell[(cell.orig_false == 0) & (cell.flipped > 0)].sort_values("flipped", ascending=False).head(15).to_string())
