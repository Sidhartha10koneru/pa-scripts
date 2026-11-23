#!/usr/bin/env bash
# run_feats.sh - attempt 5: add claim columns to the v8 passenger model, one group at a time.
# Same everything as exp_v8d_l6pax_s42 (command copied from its split_info.json: years, split, sample),
# same labels (LABEL_V6_PAX), quality codes excluded. Only --extra-feats changes, so the test services are identical.
# pa_experiments.py already supports this: --extra-feats COLS trains model "lgb_ctx_full" = 5 base features + COLS
# (Primary_DX also adds DX3 = 3-character ICD-10 category).
#
#   nohup bash run_feats.sh > run_feats.log 2>&1 &          # all groups, seed 42
#   nohup bash run_feats.sh 0 5 > run_feats.log 2>&1 &      # only groups 0 and 5
#   SEED=7 nohup bash run_feats.sh 0 > run_feats_s7.log 2>&1 &
#   bash run_feats.sh score                                  # TP/FP/FN/TN at 80/85/90% recall, all runs vs baseline
set -u
V8D="s3://cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v8_draft/ai_prior_auth_09282026_v3_{year}_std.parquet"
REF_RUN=${REF_RUN:-exp_v8d_l6pax_s42}
SEED=${SEED:-42}

#      out-name suffix   columns
GROUPS=(
  "dx        Primary_DX"
  "lob       LINE_OF_BUSINESS PRODUCT_NAME"
  "dept      SPECIALTY DEPARTMENT_ID VISITTYPE"
  "npi       SERVICEPROVIDERNPI"
  "qty       QUANTITY"
  "all       Primary_DX LINE_OF_BUSINESS PRODUCT_NAME SPECIALTY DEPARTMENT_ID VISITTYPE SERVICEPROVIDERNPI QUANTITY"
)
# NOT used, on purpose:
#   after the service / payment (leak the answer): CARC RARC LINE_AUTHORIZATION_NUMBER ACUITY_AUTH_CODES CLAIM_SUBMISSION_DATE
#                                                  CHARGEAMOUNT INVOICE* FILING_ORDER_C labels / N_* / GROUP_*
#   protected attributes: SEX ETHNICITY        patient identifiers: PAT_* MRN BIRTH_DATE PAT_ZIP ACCOUNT_ID names
#   BUCKET_* ids (not payer keys)

if [ "${1:-}" = "score" ]; then
  RUNS="base=exp_v8d_grp_s42 l6pax=exp_v8d_l6pax_s42 l6paxc=exp_v9d_l6paxc_s42"
  for g in "${GROUPS[@]}"; do
    read -r NAME _ <<< "$g"
    for s in 42 7; do [ -f "exp_feat_${NAME}_s$s/test_preds.parquet" ] && RUNS="$RUNS ${NAME}_s$s=exp_feat_${NAME}_s$s"; done
  done
  echo "scoring: $RUNS"
  python -u confusion_v9.py $RUNS --base base --recall 0.80 0.85 0.90 --out feats_confusion.csv
  exit 0
fi

CMD=$(python - "$REF_RUN" <<'EOF'
import json, re, sys
i = json.load(open(f"{sys.argv[1]}/split_info.json"))
if "command" in i:
    c = i["command"]
else:                                                      # older runs: rebuild it, as run_v9d.sh does
    y = [str(v) for v in i["years"]]
    c = " ".join(["pa_experiments.py", "baselines", "--train-years", *y, "--test-years", y[-1],
                  "--update-start", str(i["update_start"]), "--test-start", str(i["test_start"]), "--sample", str(i["sample"])]
                 + (["--test-end", str(i["test_end"])] if i.get("test_end") else [])
                 + (["--include-secondary"] if i.get("include_secondary") else [])
                 + (["--secondary-in-test"] if i.get("secondary_in_test") else []))
c = c.replace("pa_experiments.py", "pa_exp_cat.py")        # same script, ID columns forced categorical
c = re.sub(r"\s--out\s+\S+", "", c)
c = re.sub(r"\s--models(\s+[^-\s]\S*)*", "", c)          # model list is set below
c = re.sub(r"\s--extra-feats(\s+[^-\s]\S*)*", "", c)
print(c)
EOF
) || { echo "cannot read $REF_RUN/split_info.json (needs a 'command' entry)"; exit 1; }
echo "reference command (without --out/--models/--extra-feats): python -u $CMD"
mkdir -p logs_feats
SEL=("$@")
for i in "${!GROUPS[@]}"; do
  if [ ${#SEL[@]} -gt 0 ] && [[ ! " ${SEL[*]} " =~ " $i " ]]; then continue; fi
  read -r NAME COLS <<< "${GROUPS[$i]}"
  OUT="exp_feat_${NAME}_s${SEED}"
  echo "=== [$i] $OUT  extra: $COLS  $(date '+%F %T')"
  PA_BASE_PATH="$V8D" PA_LABEL_COL=LABEL_V6_PAX PA_QUALITY_MODE=exclude PA_SEED="$SEED" \
    python -u $CMD --extra-feats $COLS --models lgb_ctx_full lgb_rules_ctx_full --out "$OUT" > "logs_feats/$OUT.log" 2>&1 \
    || { echo "    FAILED - see logs_feats/$OUT.log"; tail -5 "logs_feats/$OUT.log"; continue; }
  grep -E "Primary_DX filled|trees|AP_svc" "logs_feats/$OUT.log" | tail -4
done
echo "done $(date '+%F %T')"
