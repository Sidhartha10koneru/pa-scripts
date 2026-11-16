python - <<'EOF'
import pandas as pd, glob
rows=[]
for f in sorted(glob.glob("exp_*/precision_ceiling*/metrics.csv")):
    m=pd.read_csv(f).query("scorer=='model'").iloc[0]
    rows.append({"run":f.split("/")[0],"key":"original" if "original" in f else "corrected",
                 "AUC-PR":m["AUC-PR"],"prec@80":m["precision @ recall 80%"],"FP@80":m["FP @ recall 80%"],
                 "recall@prec90":m["recall @ precision 90%"]})
print(pd.DataFrame(rows).round(3).to_string(index=False))
EOF


for r in exp_y25_tv2602_l7s_s42 exp_y25_tv2602_l6pax_s42; do echo "=== $r"; grep -iE "best_iter|iteration|early|val_lines|train_lines|error|Traceback|MISMATCH" logs_y25/$r.log | head -15; tail -5 logs_y25/$r.log; done
