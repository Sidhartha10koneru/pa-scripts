python - <<'EOF'
import pandas as pd
d = pd.read_csv("feats_confusion.csv"); d = d[d["answer key"] == "agreed"]
rows = [("Baseline (5 fields)", "l6pax", 42), ("Core (10 fields)", "core_s42", 42), ("Core (10 fields)", "core_s7", 7),
        ("Core + history", "hist_s42", 42), ("Core + history", "hist_s7", 7), ("Core + history, 30-day delay", "hist_lag30_s42", 42)]
fp = lambda m, r: int(d[(d.model == m) & (d["target recall"] == r)].FP.iloc[0])
out = []
for name, m, seed in rows:
    t = d[(d.model == m) & (d["target recall"] == "85%")].iloc[0]
    core = "core_s7" if seed == 7 else "core_s42"
    out.append({"Model": name, "Seed": seed, "FP @80%": fp(m, "80%"), "FP @85%": fp(m, "85%"),
                "Change vs baseline @85%": f"{fp(m,'85%') / fp('l6pax','85%') - 1:.1%}",
                "Change vs core @85%": f"{fp(m,'85%') / fp(core,'85%') - 1:.1%}" if m.startswith("hist") else "",
                "Precision @85%": f"{t.TP / (t.TP + t.FP):.1%}", "FP @90%": fp(m, "90%")})
T = pd.DataFrame(out); print(T.to_string(index=False)); T.to_csv("history_report_table.csv", index=False)
EOF
