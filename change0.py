python - <<'EOF'
import pandas as pd
d = pd.read_csv("feats_confusion.csv"); d = d[d["answer key"] == "agreed"]
keep = {"l6pax": "Baseline (5 fields)", "core_s42": "Core", "core_s7": "Core", "hist_s42": "Core + history",
        "hist_s7": "Core + history", "hist_lag30_s42": "Core + history, 30-day delay"}
T = d[d.model.isin(keep)].copy()
T.insert(0, "Model", T.model.map(keep)); T.insert(1, "Seed", T.model.str.extract(r"s(\d+)$")[0].fillna("42"))
T["precision"] = (T.TP / (T.TP + T.FP)).map("{:.1%}".format); T["recall"] = (T.TP / (T.TP + T.FN)).map("{:.1%}".format)
T = T.sort_values(["target recall", "model"])[["target recall", "Model", "Seed", "TP", "FP", "FN", "TN", "precision", "recall"]]
print(T.to_string(index=False)); T.to_csv("history_report_confusion.csv", index=False)
EOF
