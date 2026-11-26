#!/usr/bin/env bash
# run_rad.sh - radiology-only models (CPT 70010-79999), core feature set, one run per label.
# Same command as the core run (copied from REF_RUN's split_info.json: years, split, sample), same features
# (core = 5 base + Primary_DX SPECIALTY DEPARTMENT_ID VISITTYPE SERVICEPROVIDERNPI), quality codes excluded.
# Only two things change: lines outside radiology are dropped after loading (pa_exp_rad.py), and the label.
#
#   nohup bash run_rad.sh > run_rad.log 2>&1 &                      # both labels, seed 42
#   LABELS="LABEL_V7_STRICT" nohup bash run_rad.sh > run_rad.log 2>&1 &
#   python rad_score.py ...                                         # scoring, see rad_score.py
set -u
V8D=${BASE_PATH:-"s3://cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v8_draft/ai_prior_auth_09282026_v3_{year}_std.parquet"}
REF_RUN=${REF_RUN:-exp_feat_core_s42}
SEED=${SEED:-42}
LABELS=${LABELS:-"LABEL_V6_PAX LABEL_V7_STRICT"}
RANGE=${PA_CPT_RANGE:-"70010 79999"}
CORE="Primary_DX SPECIALTY DEPARTMENT_ID VISITTYPE SERVICEPROVIDERNPI"

CMD=$(python - "$REF_RUN" <<'EOF'
import json, re, sys
c = json.load(open(f"{sys.argv[1]}/split_info.json"))["command"]
c = re.sub(r"\S*pa_exp(eriments|_cat)\.py", "pa_exp_rad.py", c)
c = re.sub(r"\s--out\s+\S+", "", c)
c = re.sub(r"\s--models(\s+[^-\s]\S*)*", "", c)
c = re.sub(r"\s--extra-feats(\s+[^-\s]\S*)*", "", c)
print(c)
EOF
) || { echo "cannot read $REF_RUN/split_info.json"; exit 1; }
echo "reference command: python -u $CMD"
mkdir -p logs_rad
for L in $LABELS; do
  SHORT=$(echo "$L" | sed 's/^LABEL_//' | tr 'A-Z' 'a-z')
  OUT="exp_rad_${SHORT}_s${SEED}"
  echo "=== $OUT  label $L  cpt $RANGE  $(date '+%F %T')"
  PA_BASE_PATH="$V8D" PA_LABEL_COL="$L" PA_QUALITY_MODE=exclude PA_SEED="$SEED" PA_CPT_RANGE="$RANGE" \
    python -u $CMD --extra-feats $CORE --models lgb_ctx_full --out "$OUT" > "logs_rad/$OUT.log" 2>&1 \
    || { echo "    FAILED - see logs_rad/$OUT.log"; tail -5 "logs_rad/$OUT.log"; continue; }
  grep -E "CPT range|TRAIN |UPDATE |TEST |trees|AP_svc" "logs_rad/$OUT.log" | tail -8
done
echo "done $(date '+%F %T')"
