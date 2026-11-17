python - <<'EOF'
import pyarrow.dataset as ds, pyarrow.fs as pafs, pandas as pd, os
fs = pafs.S3FileSystem()
p = "cli-ml-data-prod/ai-prior-auth/accureg_de_id_data/UABHS_April2026_scrubbed_YOBretained.parquet"
d = ds.dataset(p, filesystem=fs, format="parquet")
t = d.head(50000).to_pandas()
# standard HL7 v2 meanings for the common fields (AccuReg may use some differently)
STD = {"MSH_6":"message date/time (MSH-7)","MSH_8":"message type (MSH-9)","MSH_9":"control id (MSH-10)","MSH_11":"version (MSH-12)",
 "EVN_1":"event type","EVN_2":"event date/time","PV1_2":"patient class","PV1_3":"assigned location","PV1_4":"admission type",
 "PV1_7":"attending doctor","PV1_8":"referring doctor","PV1_9":"consulting doctor","PV1_10":"hospital service","PV1_14":"admit source",
 "PV1_17":"admitting doctor","PV1_18":"patient type","PV1_19":"visit number","PV1_20":"financial class","PV1_36":"discharge disposition",
 "PV1_39":"servicing facility","PV1_44":"admit date/time","PV1_45":"discharge date/time","PV2_3":"admit reason","PV2_8":"expected admit date",
 "PID_7":"date of birth","PID_8":"sex","PID_10":"race","PID_22":"ethnic group","DG1":"diagnosis (list)","PR1":"procedure (list)",
 "IN1":"insurance (list)","IN2":"insurance additional info (list)","IN3":"insurance certification (list)","GT1":"guarantor (list)",
 "NK1":"next of kin (list)","ACC":"accident","PD1_4":"primary care provider"}
PERSONAL = ("PID", "PD1", "NK1", "GT1", "IN2")
DOCTORS = {"PV1_7", "PV1_8", "PV1_9", "PV1_17", "PD1_4"}
rows = []
for c in t.columns:
    s = t[c].astype(str)
    s = s[~s.isin(["", "None", "nan", '""', "[]"])]
    seg = c.split("_")[0]
    safe = not seg.startswith(PERSONAL) and c not in DOCTORS and len(s) and s.str.len().median() < 60
    rows.append({"column": c, "segment": seg, "standard_meaning": STD.get(c, ""), "filled_%": round(100 * len(s) / len(t), 1),
                 "distinct": s.nunique(), "examples": " | ".join(s.value_counts().head(3).index.str.slice(0, 25)) if safe else ""})
R = pd.DataFrame(rows)
os.makedirs("accureg_profile", exist_ok=True)
R.to_csv("accureg_profile/columns_inventory.csv", index=False)
print(f"rows read: {len(t):,} of {d.count_rows():,} | columns: {len(R)} | empty columns: {(R['filled_%'] == 0).sum()}")
print(R[R["filled_%"] > 0].sort_values(["segment", "filled_%"], ascending=[True, False]).to_string(index=False))
os._exit(0)
EOF
