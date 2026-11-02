x = x.sort_values(["PAT_ID", "SERVICEDATE"])
x["dx3"] = x.Primary_DX.astype(str).str[:3]
x["new_dx_for_pat_spec"] = ~x.duplicated(["PAT_ID", "SPECIALTY", "dx3"])     # first time this patient+specialty+dx category appears
x["new_spec_for_pat"] = ~x.duplicated(["PAT_ID", "SPECIALTY"])
for c in ["new_dx_for_pat_spec", "new_spec_for_pat"]:
    print(x.groupby(c).y.agg(lines="size", pa="sum", pa_rate="mean"))
