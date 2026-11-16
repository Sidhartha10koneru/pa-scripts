python - <<'EOF'
import pandas as pd
z = pd.read_csv("pax_zero_pa_cpt_payer.csv").assign(rate=lambda x: x.pa_before / x.lines)
for t in (0.10, 0.20, 0.50):
    s = z[(z.rate >= t) & (z.lines >= 20)]
    print(f"rate >= {t:.0%}, >= 20 lines: {len(s)} combos, {int(s.pa_before.sum()):,} PA lines")
print(z[(z.rate >= 0.20) & (z.lines >= 20)].sort_values("pa_before", ascending=False).head(30).to_string(index=False))
EOF
