python - <<'EOF'
import pandas as pd
s = pd.read_parquet("pax_cpt_payer_summary.parquet")
f = s[s.flipped > 0]
print(f.groupby("payor")[["flipped","pa_before"]].sum().sort_values("flipped", ascending=False).head(15).to_string())
print(f.sort_values("flipped", ascending=False)[["cpt","payor","pa_before","flipped","pa_after"]].head(30).to_string(index=False))
f.sort_values("flipped", ascending=False).to_csv("pax_flipped_by_cpt_payer.csv", index=False)
EOF
