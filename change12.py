python - <<'EOF'
import pandas as pd, glob, os
pd.set_option("display.width", 220)
for s in ["s42", "s7"]:
    d = f"v9d_compare_{s}"
    print(f"\n######## seed {s.upper()} ########")
    for f in ["matrix_service.csv", "agree.csv", "pax.csv"]:
        p = f"{d}/{f}"
        if os.path.exists(p):
            print(f"\n== {f}"); print(pd.read_csv(p).round(4).to_string(index=False))
    p = f"v9d_optc_check_{s}.csv"
    if os.path.exists(p):
        print(f"\n== {p}"); print(pd.read_csv(p).to_string(index=False))
rows = []
for f in sorted(glob.glob("exp_v*_l6pax*/precision_ceiling*/metrics.csv")):
    m = pd.read_csv(f).query("scorer == 'model'").iloc[0]
    rows.append({"run": f.split("/")[0], "key": f.split("/")[1].replace("precision_ceiling", "corrected").replace("corrected_", ""),
                 "AUC-PR": m["AUC-PR"], "prec@80": m["precision @ recall 80%"], "FP@80": m["FP @ recall 80%"]})
print("\n== precision at 80% catch, by answer key"); print(pd.DataFrame(rows).round(3).to_string(index=False))
EOF
