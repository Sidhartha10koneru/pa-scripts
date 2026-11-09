import pandas as pd, pyarrow.dataset as ds, pyarrow.fs as pafs
fs = pafs.S3FileSystem()
P = "cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v8_draft/ai_prior_auth_09282026_v3_{y}_std.parquet"
COLS = ["FINAL_CPT", "GROUP_LABEL", "LABEL_V6", "LABEL_V7", "LABEL_V7_STRICT", "IS_PASSENGER", "VISIT_LEVEL_AUTH", "IS_QUALITY_CODE"]
U = lambda s: s.astype(str).str.strip().str.upper().replace({"1": "TRUE", "1.0": "TRUE", "0": "FALSE", "0.0": "FALSE"})
plist = set(pd.read_csv("v6_impact_owner/P_list_thr2.csv", dtype=str).FINAL_CPT)
cpt_parts, move_parts = [], []
for y in [2022, 2023, 2024, 2025, 2026]:
    for b in ds.dataset(P.format(y=y), filesystem=fs).to_batches(columns=COLS, batch_size=1_000_000):
        t = b.to_pandas()
        for c in ["GROUP_LABEL", "LABEL_V6", "LABEL_V7", "LABEL_V7_STRICT"]:
            t[c] = U(t[c]).where(lambda s: s.isin(["TRUE", "FALSE"]), "AMBIGUOUS")
        t["FINAL_CPT"] = t.FINAL_CPT.astype(str).str.strip().str.upper()
        # (a) v6 -> v7: where did the new TRUE come from
        move_parts.append(t.groupby(["LABEL_V6", "LABEL_V7"]).size().rename("lines").reset_index().assign(label="LABEL_V7", year=y))
        move_parts.append(t.groupby(["LABEL_V6", "LABEL_V7_STRICT"]).size().rename("lines").reset_index()
                          .rename(columns={"LABEL_V7_STRICT": "LABEL_V7"}).assign(label="LABEL_V7_STRICT", year=y))
        # (b) per passenger-list CPT
        s = t[t.FINAL_CPT.isin(plist)]
        cpt_parts.append(s.groupby("FINAL_CPT").agg(
            lines=("LABEL_V6", "size"), true_v6=("LABEL_V6", lambda x: (x == "TRUE").sum()),
            passengers=("IS_PASSENGER", "sum"), visit_level_kept=("VISIT_LEVEL_AUTH", "sum")).reset_index())
M = pd.concat(move_parts).groupby(["label", "LABEL_V6", "LABEL_V7"]).lines.sum().unstack(fill_value=0)
print("== v6 -> v7 label moves (raw lines, 2022-26)\n", M.to_string())
C = pd.concat(cpt_parts).groupby("FINAL_CPT").sum()
C["true_kept"] = C.true_v6 - C.passengers
C["share_flipped"] = (C.passengers / C.true_v6.replace(0, pd.NA)).round(3)
A = pd.read_csv("v6_impact_owner/alone_rates_by_cpt.csv", dtype={"FINAL_CPT": str}).set_index("FINAL_CPT")
C = C.join(A[["size_alone", "rate_alone", "size_with", "rate_with"]])
C = C.sort_values("passengers", ascending=False)
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 300)
print("\n== per CPT on the passenger list\n", C.head(40).round(4).to_string())
C.to_csv("passenger_by_cpt.csv"); M.to_csv("v6_to_v7_moves.csv")
