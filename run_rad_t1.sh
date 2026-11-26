#!/usr/bin/env bash
# run_rad.sh - radiology-only models (CPT 70010-79999), core feature set, one run per label.
# Same command as the core run (copied from REF_RUN's split_info.json: years, split, sample), same features
# (core = 5 base + Primary_DX SPECIALTY DEPARTMENT_ID VISITTYPE SERVICEPROVIDERNPI), quality codes excluded.
# Only two things change: lines outside radiology are dropped after loading (pa_exp_rad.py), and the label.
#
#   nohup bash run_rad.sh > run_rad.log 2>&1 &                      # both labels, seed 42
#   LABELS="LABEL_V7_STRICT" nohup bash run_rad.sh > run_rad.log 2>&1 &
#   TRAIN_YEARS="2025 2026" nohup bash run_rad.sh > run_rad_y2025.log 2>&1 &   # -> exp_rad_<label>_y2025_s42
#   EXTRA_ARGS="--es-metric average_precision --val-start 2026-01-01 --refit-full" RUN_TAG=esap \
#     nohup bash run_rad.sh > run_rad_esap.log 2>&1 &                        # early stopping on AUC-PR, time-based validation
#   python rad_score.py ...                                         # scoring, see rad_score.py
set -u
V8D=${BASE_PATH:-"s3://cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v8_draft/ai_prior_auth_09282026_v3_{year}_std.parquet"}
REF_RUN=${REF_RUN:-exp_feat_core_s42}
SEED=${SEED:-42}
LABELS=${LABELS:-"LABEL_V6_PAX LABEL_V7_STRICT"}
RANGE=${PA_CPT_RANGE:-"70010 79999"}
CORE="Primary_DX SPECIALTY DEPARTMENT_ID VISITTYPE SERVICEPROVIDERNPI"

CMD=$(python - "$REF_RUN" <<'EOF'
import json, os, re, sys
c = json.load(open(f"{sys.argv[1]}/split_info.json"))["command"]
c = re.sub(r"\S*pa_exp(eriments|_cat)\.py", "pa_exp_rad.py", c)
c = re.sub(r"\s--out\s+\S+", "", c)
c = re.sub(r"\s--models(\s+[^-\s]\S*)*", "", c)
c = re.sub(r"\s--extra-feats(\s+[^-\s]\S*)*", "", c)
if os.environ.get("TRAIN_YEARS"):                          # e.g. TRAIN_YEARS="2025 2026": recent years only
    c = re.sub(r"--train-years(\s+\d{4})+", "--train-years " + os.environ["TRAIN_YEARS"], c)
print(c)
EOF
) || { echo "cannot read $REF_RUN/split_info.json"; exit 1; }
echo "reference command: python -u $CMD"
mkdir -p logs_rad
for L in $LABELS; do
  SHORT=$(echo "$L" | sed 's/^LABEL_//' | tr 'A-Z' 'a-z')
  TAG=""; [ -n "${TRAIN_YEARS:-}" ] && TAG="_y$(echo $TRAIN_YEARS | awk '{print $1}')"
  [ -n "${RUN_TAG:-}" ] && TAG="${TAG}_${RUN_TAG}"          # e.g. RUN_TAG=esap -> exp_rad_v6_pax_esap_s42
  OUT="exp_rad_${SHORT}${TAG}_s${SEED}"
  echo "=== $OUT  label $L  cpt $RANGE  $(date '+%F %T')"
  PA_BASE_PATH="$V8D" PA_LABEL_COL="$L" PA_QUALITY_MODE=exclude PA_SEED="$SEED" PA_CPT_RANGE="$RANGE" \
    python -u $CMD --extra-feats $CORE --models lgb_ctx_full ${EXTRA_ARGS:-} --out "$OUT" > "logs_rad/$OUT.log" 2>&1 \
    || { echo "    FAILED - see logs_rad/$OUT.log"; tail -5 "logs_rad/$OUT.log"; continue; }
  grep -E "CPT range|TRAIN |UPDATE |TEST |trees|AP_svc" "logs_rad/$OUT.log" | tail -8
done
echo "done $(date '+%F %T')"


EXTRA_ARGS="--es-metric average_precision --val-start 2026-01-01 --refit-full" RUN_TAG=esap \
  nohup bash run_rad.sh > run_rad_esap.log 2>&1 &


python rad_score.py core=exp_feat_core_s42 rad_v6pax=exp_rad_v6_pax_s42 rad_v7strict=exp_rad_v7_strict_s42 \
  rad_v6pax_esap=exp_rad_v6_pax_esap_s42 rad_v7strict_esap=exp_rad_v7_strict_esap_s42 --out rad_scores.csv
