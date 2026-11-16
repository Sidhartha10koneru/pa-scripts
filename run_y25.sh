#!/usr/bin/env bash
# run_y25.sh - train on 2025 - Apr 2026 only, for two labels, then register + cut-off check + precision ceiling.
#   labels:  LABEL_V6_PAX     quality codes excluded + passenger fix (current best)
#            LABEL_V7_STRICT  quality codes excluded + Acuity auth-only denial codes, NO passenger fix
#   validation:  rv10 = random 10% of visits | tv2602 = time-based, Feb-Apr 2026
# Same settings as run_window_val.sh; a run whose folder already has test_preds.parquet is NOT retrained (only re-checked).
#
#   nohup bash run_y25.sh 0 1 2 3 > run_y25.log 2>&1 &     # seed 42 first
#   nohup bash run_y25.sh 4 5 6 7 > run_y25_s7.log 2>&1 &  # seed 7 check, later
#   bash run_y25.sh 8                                     # optional: V7 strict on ALL years (to compare with exp_v8d_l6pax_s42)
set -u
V8D="s3://cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v8_draft/ai_prior_auth_09282026_v3_{year}_std.parquet"
FROZEN=frozen/test_v1_2026-05_2026-07
COMMON="--test-years 2026 --update-start 2026-02-01 --test-start 2026-05-01 --test-end 2026-08-01 --sample 1.0 --models lgb_full --include-secondary --secondary-in-test"
#     out                        label             train years                val-start   seed
RUNS=(
  "exp_y25_rv10_l6pax_s42      LABEL_V6_PAX      2025_2026                  -           42"
  "exp_y25_tv2602_l6pax_s42    LABEL_V6_PAX      2025_2026                  2026-02-01  42"
  "exp_y25_rv10_l7s_s42        LABEL_V7_STRICT   2025_2026                  -           42"
  "exp_y25_tv2602_l7s_s42      LABEL_V7_STRICT   2025_2026                  2026-02-01  42"
  "exp_y25_rv10_l6pax_s7       LABEL_V6_PAX      2025_2026                  -           7"
  "exp_y25_tv2602_l6pax_s7     LABEL_V6_PAX      2025_2026                  2026-02-01  7"
  "exp_y25_rv10_l7s_s7         LABEL_V7_STRICT   2025_2026                  -           7"
  "exp_y25_tv2602_l7s_s7       LABEL_V7_STRICT   2025_2026                  2026-02-01  7"
  "exp_v8d_l7s_s42             LABEL_V7_STRICT   2022_2023_2024_2025_2026   -           42"
)
mkdir -p logs_y25
SEL=("$@")
for i in "${!RUNS[@]}"; do
  if [ ${#SEL[@]} -gt 0 ] && [[ ! " ${SEL[*]} " =~ " $i " ]]; then continue; fi
  read -r OUT LABEL YEARS VS SEED <<< "${RUNS[$i]}"
  YRS=${YEARS//_/ }; VAL=""; [ "$VS" != "-" ] && VAL="--val-start $VS"
  LOG="logs_y25/$OUT.log"
  echo "=== [$i] $OUT  label=$LABEL  years=$YRS  val=$VS  seed=$SEED  $(date '+%F %T')"
  if [ -f "$OUT/test_preds.parquet" ]; then
    echo "    already trained - skipping training"
  else
    PA_BASE_PATH="$V8D" PA_LABEL_COL=$LABEL PA_QUALITY_MODE=exclude PA_SEED=$SEED \
      python -u pa_experiments.py baselines --train-years $YRS $COMMON $VAL --out "$OUT" > "$LOG" 2>&1 \
      || { echo "    FAILED - see $LOG"; tail -5 "$LOG"; continue; }
    # older pa_experiments.py does not record its settings: add them so the run ID is right
    python - "$OUT" "$V8D" "$LABEL" "$SEED" <<'PY'
import json, sys
out, bp, lab, seed = sys.argv[1:]
p = f"{out}/split_info.json"; i = json.load(open(p))
if "label_col" not in i:
    i.update(base_path=bp, label_col=lab, quality_mode="exclude", seed=int(seed), provenance_added_by="run_y25.sh")
    json.dump(i, open(p, "w"), indent=1)
PY
    python register_run.py "$OUT" --frozen "$FROZEN" --status candidate --notes "2025-only training / label check" >> "$LOG" 2>&1 \
      && grep -E "^pa-|AUC_PR|FP@80|coverage" "$LOG" | tail -5
  fi
  # cut-off chosen on validation -> achieved on frozen test (original + passenger-corrected answer keys)
  python -u cutoff_check.py --run "$OUT" --seed $SEED --label $LABEL --quality-mode exclude --base-path "$V8D" \
    --years $YRS $VAL >> "$LOG" 2>&1 && grep -A 13 "^== $OUT" "$LOG" | tail -14
  # same frozen services, both answer keys -> comparable with every other run
  for KEY in exp_v8d_l6pax_s42 none; do
    if [ "$KEY" = none ]; then LF=""; OD="$OUT/precision_ceiling_original"; else LF="--labels-from $KEY"; OD="$OUT/precision_ceiling"; fi
    python precision_ceiling.py --run "$OUT" --frozen "$FROZEN" $LF --out "$OD" >> "$LOG" 2>&1 \
      && echo "    precision @ 80% catch ($([ "$KEY" = none ] && echo original || echo corrected) labels):" \
      && python -c "import pandas as p; r=p.read_csv('$OD/metrics.csv'); print(r[['scorer','AUC-PR','precision @ recall 80%','FP @ recall 80%','recall @ precision 90%']].round(3).to_string(index=False))"
  done
done
echo "done $(date '+%F %T')"
