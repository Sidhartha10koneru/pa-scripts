#!/usr/bin/env bash
# run_v8d.sh - train the v8-draft label candidates with EXACTLY the settings of the quality-exclude candidate run,
# then register each on the frozen official test set (original labels -> every run scored on the same services/labels).
#
# Only PA_BASE_PATH, PA_LABEL_COL and PA_SEED change; the training command is copied from REF_RUN/split_info.json.
# Runs one at a time (memory). Order = most useful first, so you can stop early.
#
#   bash run_v8d.sh                 # all runs
#   bash run_v8d.sh 0 1             # only runs 0 and 1 (see RUNS below)
# Logs: logs_v8d/<out>.log | registry: registry.csv (local; push later once the S3 location is agreed)
set -u
REF_RUN=${REF_RUN:-exp_qc_exclude_rv}                  # registered candidate: pa-lgb5-v4-grp-qcexcl-rv10-s42-...
FROZEN=${FROZEN:-frozen/test_v1_2026-05_2026-07}
V8D="s3://cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v8_draft/ai_prior_auth_09282026_v3_{year}_std.parquet"

#      out folder              label column           seed  status      note
RUNS=(
  "exp_v8d_grp_s42           GROUP_LABEL            42    archived  sanity: v8 files reproduce the exclude candidate (expect AUC-PR ~0.6527)"
  "exp_v8d_l7spax_s42        LABEL_V7_STRICT_PAX    42    candidate recommended: Acuity auth-only codes + passenger fix"
  "exp_v8d_l6pax_s42         LABEL_V6_PAX           42    candidate passenger fix only"
  "exp_v8d_l7pax_s42         LABEL_V7_PAX           42    candidate full Acuity list + passenger fix (changes the target)"
  "exp_v8d_l7spax_s7         LABEL_V7_STRICT_PAX    7     candidate seed check"
  "exp_v8d_l6pax_s7          LABEL_V6_PAX           7     candidate seed check"
  "exp_v8d_l7pax_s7          LABEL_V7_PAX           7     candidate seed check"
)

CMD=$(python - "$REF_RUN" <<'EOF'
import json, re, sys
i = json.load(open(f"{sys.argv[1]}/split_info.json"))
if "command" in i:                       # run made after provenance was added: exact command
    print(re.sub(r"\s--out\s+\S+", "", i["command"]))
    sys.exit(0)
# older run: rebuild the command from the settings it recorded (same split, filters, models)
if i.get("rules"):
    sys.exit("reference run used the rules feature - its command cannot be rebuilt; pass a REF_RUN with 'command'")
y = [str(v) for v in i["years"]]
c = ["pa_experiments.py", "baselines", "--train-years", *y, "--test-years", y[-1],
     "--update-start", str(i["update_start"]), "--test-start", str(i["test_start"]), "--sample", str(i["sample"])]
if i.get("test_end"):          c += ["--test-end", str(i["test_end"])]
if i.get("models"):            c += ["--models", *i["models"]]
if i.get("include_secondary"): c += ["--include-secondary"]
if i.get("secondary_in_test"): c += ["--secondary-in-test"]
if i.get("dedup_keep_true") is False: c += ["--dedup-first"]
if i.get("val_start"):         c += ["--val-start", str(i["val_start"])]
if i.get("refit_full"):        c += ["--refit-full"]
if i.get("es_metric"):         c += ["--es-metric", str(i["es_metric"])]
print(" ".join(c))
print("NOTE: rebuilt from recorded settings (no 'command' in split_info.json) - check it matches the run", file=sys.stderr)
EOF
) || { echo "cannot read $REF_RUN/split_info.json"; exit 1; }
echo "reference command (without --out): python -u $CMD"
mkdir -p logs_v8d

SEL=("$@")
for i in "${!RUNS[@]}"; do
  if [ ${#SEL[@]} -gt 0 ] && [[ ! " ${SEL[*]} " =~ " $i " ]]; then continue; fi
  read -r OUT LABEL SEED STATUS NOTE <<< "${RUNS[$i]}"
  echo "=== [$i] $OUT  label=$LABEL seed=$SEED  $(date '+%F %T')"
  PA_BASE_PATH="$V8D" PA_LABEL_COL="$LABEL" PA_QUALITY_MODE=exclude PA_SEED="$SEED" \
    python -u $CMD --out "$OUT" > "logs_v8d/$OUT.log" 2>&1
  if [ $? -ne 0 ]; then echo "    FAILED - see logs_v8d/$OUT.log"; tail -5 "logs_v8d/$OUT.log"; continue; fi
  # older pa_experiments.py does not record its settings: add what this script set, so the run ID is right
  python - "$OUT" "$V8D" "$LABEL" "$SEED" "python -u $CMD --out $OUT" <<'EOF'
import json, sys
out, bp, lab, seed, cmd = sys.argv[1:]
p = f"{out}/split_info.json"; i = json.load(open(p))
if "label_col" not in i:
    i.update(base_path=bp, label_col=lab, quality_mode="exclude", seed=int(seed), command=cmd,
             env={"PA_BASE_PATH": bp, "PA_LABEL_COL": lab, "PA_QUALITY_MODE": "exclude", "PA_SEED": seed},
             provenance_added_by="run_v8d.sh")
    json.dump(i, open(p, "w"), indent=1)
    print("    settings added to split_info.json (pa_experiments.py has no provenance recording)")
EOF
  python register_run.py "$OUT" --frozen "$FROZEN" --status "$STATUS" --notes "v8 draft: $NOTE" \
    >> "logs_v8d/$OUT.log" 2>&1 && grep -E "^pa-|AUC_PR|FP@80|coverage" "logs_v8d/$OUT.log" | tail -6
done
echo "done $(date '+%F %T')"


# nohup bash run_v8d.sh > run_v8d.log 2>&1 &        # all 7
# nohup bash run_v8d.sh 0 1 > run_v8d.log 2>&1 &    # or just sanity + recommended first
# tail -f run_v8d.log
