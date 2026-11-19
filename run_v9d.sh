#!/usr/bin/env bash
# run_v9d.sh - option C passenger labels (LABEL_V6_PAXC): train, register, compare with the current passenger model.
# Same training settings as the v8 runs (command copied from REF_RUN), all years, quality codes excluded.
#
# 0. build the v9 files first (build_v9_optc.py), then set V9D below to the SAME agreed S3 folder.
#    nohup bash run_v9d.sh 0 1 > run_v9d.log 2>&1 &      # train seed 42 and 7
#    nohup bash run_v9d.sh compare > run_v9d_compare.log 2>&1 &
#    bash run_v9d.sh 2                                       # optional sanity: LABEL_V6_PAX on v9 files = v8 result
set -u
V9D=${V9D:-"s3://<AGREED LOCATION>/std_data_v9_draft/ai_prior_auth_09282026_v3_{year}_std.parquet"}
V8D="s3://cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v8_draft/ai_prior_auth_09282026_v3_{year}_std.parquet"
REF_RUN=${REF_RUN:-exp_qc_exclude_rv}
FROZEN=${FROZEN:-frozen/test_v1_2026-05_2026-07}
[[ "$V9D" == *"<AGREED"* ]] && { echo "set V9D to the agreed v9 folder first (export V9D=...)"; exit 1; }

#     out folder              label column     seed  status     note
RUNS=(
  "exp_v9d_l6paxc_s42      LABEL_V6_PAXC    42    candidate  option C passenger labels"
  "exp_v9d_l6paxc_s7       LABEL_V6_PAXC    7     candidate  option C passenger labels, seed check"
  "exp_v9d_l6pax_s42       LABEL_V6_PAX     42    archived   sanity: v9 files reproduce exp_v8d_l6pax_s42"
)

# register_run.py: short tag for the new label (once)
grep -q LABEL_V6_PAXC register_run.py || \
  sed -i 's/"LABEL_V6_PAX": "l6pax",/"LABEL_V6_PAX": "l6pax", "LABEL_V6_PAXC": "l6paxc",/' register_run.py

if [ "${1:-}" = "compare" ]; then
  python -u compare_v8d.py base=exp_v8d_grp_s42 l6pax=exp_v8d_l6pax_s42 l6paxc=exp_v9d_l6paxc_s42 \
    --base base --out v9d_compare_s42 --v8-path "$V8D"
  python -u compare_v8d.py base=exp_exclude_s7 l6pax=exp_v8d_l6pax_s7 l6paxc=exp_v9d_l6paxc_s7 \
    --base base --out v9d_compare_s7 --v8-path "$V8D"
  python -u optc_check.py base=exp_v8d_grp_s42 l6pax=exp_v8d_l6pax_s42 l6paxc=exp_v9d_l6paxc_s42 --base base \
    --v9-path "$V9D" --out v9d_optc_check_s42.csv
  python -u optc_check.py base=exp_exclude_s7 l6pax=exp_v8d_l6pax_s7 l6paxc=exp_v9d_l6paxc_s7 --base base \
    --v9-path "$V9D" --out v9d_optc_check_s7.csv
  for r in exp_v9d_l6paxc_s42 exp_v9d_l6paxc_s7; do         # same frozen services, both answer keys + the new one
    python precision_ceiling.py --run $r --frozen "$FROZEN" --labels-from exp_v8d_l6pax_s42 --out $r/precision_ceiling
    python precision_ceiling.py --run $r --frozen "$FROZEN" --out $r/precision_ceiling_original
    python precision_ceiling.py --run $r --frozen "$FROZEN" --labels-from exp_v9d_l6paxc_s42 --out $r/precision_ceiling_paxc
  done
  echo "done $(date '+%F %T')"; exit 0
fi

CMD=$(python - "$REF_RUN" <<'EOF'
import json, re, sys
i = json.load(open(f"{sys.argv[1]}/split_info.json"))
if "command" in i:
    print(re.sub(r"\s--out\s+\S+", "", i["command"])); sys.exit(0)
y = [str(v) for v in i["years"]]
c = ["pa_experiments.py", "baselines", "--train-years", *y, "--test-years", y[-1],
     "--update-start", str(i["update_start"]), "--test-start", str(i["test_start"]), "--sample", str(i["sample"])]
if i.get("test_end"):          c += ["--test-end", str(i["test_end"])]
if i.get("models"):            c += ["--models", *i["models"]]
if i.get("include_secondary"): c += ["--include-secondary"]
if i.get("secondary_in_test"): c += ["--secondary-in-test"]
print(" ".join(c))
EOF
) || { echo "cannot read $REF_RUN/split_info.json"; exit 1; }
echo "training command (without --out): python -u $CMD"
mkdir -p logs_v9d
SEL=("$@")
for i in "${!RUNS[@]}"; do
  if [ ${#SEL[@]} -gt 0 ] && [[ ! " ${SEL[*]} " =~ " $i " ]]; then continue; fi
  read -r OUT LABEL SEED STATUS NOTE <<< "${RUNS[$i]}"
  echo "=== [$i] $OUT  label=$LABEL seed=$SEED  $(date '+%F %T')"
  PA_BASE_PATH="$V9D" PA_LABEL_COL="$LABEL" PA_QUALITY_MODE=exclude PA_SEED="$SEED" \
    python -u $CMD --out "$OUT" > "logs_v9d/$OUT.log" 2>&1 \
    || { echo "    FAILED - see logs_v9d/$OUT.log"; tail -5 "logs_v9d/$OUT.log"; continue; }
  python - "$OUT" "$V9D" "$LABEL" "$SEED" "python -u $CMD --out $OUT" <<'EOF'
import json, sys
out, bp, lab, seed, cmd = sys.argv[1:]
p = f"{out}/split_info.json"; i = json.load(open(p))
if "label_col" not in i:
    i.update(base_path=bp, label_col=lab, quality_mode="exclude", seed=int(seed), command=cmd, provenance_added_by="run_v9d.sh")
    json.dump(i, open(p, "w"), indent=1)
EOF
  python register_run.py "$OUT" --frozen "$FROZEN" --status "$STATUS" --notes "v9 draft: $NOTE" \
    >> "logs_v9d/$OUT.log" 2>&1 && grep -E "^pa-|AUC_PR|FP@80|coverage" "logs_v9d/$OUT.log" | tail -6
done
echo "done $(date '+%F %T')"
