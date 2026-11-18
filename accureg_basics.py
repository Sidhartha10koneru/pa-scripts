import pandas as pd, pyarrow.dataset as ds, pyarrow.fs as pafs
fs = pafs.S3FileSystem()
d = ds.dataset("cli-ml-data-prod/ai-prior-auth/accureg_de_id_data/UABHS_April2026_scrubbed_YOBretained.parquet",
               filesystem=fs, format="parquet")
print(d.count_rows(), "rows,", len(d.schema.names), "columns")
df = d.head(100_000, columns=["MSH_10", "EVN_1", "PV1_2", "PV1_10", "PV1_20", "PV2_3", "IN1", "IN3"]).to_pandas()


pd.Series(d.schema.names).str.split("_").str[0].value_counts()

df["MSH_10"].value_counts()          # message type
df["PV1_2"].value_counts()           # patient class
df["PV2_3"].value_counts().head(20)  # admit reason

df["IN1"].dropna().iloc[0]

in1 = pd.DataFrame([x for v in df["IN1"].dropna() for x in v])
in1.columns.tolist()

in1["IN1_2"].value_counts().head(20)   # plan / company code
in1["IN1_4"].value_counts().head(20)   # company name
in1.groupby(["IN1_2", "IN1_4"]).size().sort_values(ascending=False).head(20)   # code -> name

in3 = pd.DataFrame([x for v in df["IN3"].dropna() for x in v])
in3["IN3_2"].value_counts().head(20)

Step 5 assumes IN1 holds real lists. If it comes back as text instead (it would print as one long string starting with [{), add import ast and use ast.literal_eval(v) in place of v.
Stick to the columns above when showing values. Avoid PID_*, NK1, GT1, IN2, and IN1 fields 16, 18, 19, 36 and 49 (insured name, birth date, address, policy number, insured ID). The data is de-identified, but there’s no need to put them on screen.
Step 7 shows raw auth-number values, so keep the notebook on EFS and don’t share its output outside the team.
