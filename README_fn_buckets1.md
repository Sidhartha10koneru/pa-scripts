# FN buckets: why the model misses PA, tracked run by run

`fn_buckets.py` sorts every missed PA service (false negative) of a trained model into 5 buckets and appends the counts to a shared history file, so each new training run shows whether the buckets are shrinking. It is the script form of `fn_bucket_analysis.py` (cells 1–8), with the same definitions.

## Run it after every training run

It is self-contained: it needs only Python with `pandas`, `numpy` and `pyarrow` (plus S3 access to the claims files). It does the same data preparation as the training pipeline itself, so no project scripts are needed. Then:

```bash
# recommended: every run scored on the same services and the same answer key
python fn_buckets.py --run <run folder> \
    --frozen frozen/test_v1_2026-05_2026-07 --labels-from exp_v8d_l6pax_s42

# see all runs so far
python fn_buckets.py --history
```

A run folder without `split_info.json` (any model, any pipeline), or a run made before settings were recorded, needs the data settings on the command line, e.g. the original benchmark:

```bash
python fn_buckets.py --run exp_nofilter_2022 --label GROUP_LABEL --quality-mode keep \
    --base-path "s3://cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v4/ai_prior_auth_09282026_v3_{year}_std.parquet" \
    --years 2022 2023 2024 2025 2026 --test-start 2026-05-01 \
    --frozen frozen/test_v1_2026-05_2026-07 --labels-from exp_v8d_l6pax_s42
```

## Inputs

| Input | What it is | Where it comes from |
|---|---|---|
| `--run` | The model's run folder | Needs a test-predictions file (below); `split_info.json` if present |
| Test predictions | `test_preds.parquet` in the run folder (or `--preds <file>`): one row per **test line** | Columns `CSN`, `FINAL_CPT`, `FINAL_MODIFIERS`, `PLACE_OF_SERVICE_CODE`, `PAYOR_PLAN_CLEAN` (or `PAYOR` + `PLAN`), `y` (label 0/1) and the score (a `p_*` column, or name it with `--score-col`) |
| Training data settings | Claims file path (with `{year}`), label column, quality-code mode, years, test start, secondary claims | Read from the run's `split_info.json`; any can be given or overridden: `--base-path --label --quality-mode --years --test-start --include-secondary` |
| `--frozen` (recommended) | The frozen official test set | `frozen/test_v1_2026-05_2026-07` (quality codes excluded) |
| `--labels-from` (recommended) | The run whose test labels are the answer key | e.g. `exp_v8d_l6pax_s42` = corrected labels (passengers not PA) |
| `--recall` `--min-n` `--low` `--high` | Cut-off target and bucket thresholds | Defaults 0.80, 20, 0.05, 0.50 |
| `--cutoff` (optional) | A fixed score cut-off instead of a recall target, e.g. `0.8` | Then total FN also moves between runs |

Without `--frozen` / `--labels-from` each run is scored on its own test set and its own labels. Counts then change partly because the labels changed, not the model, so use them when comparing runs.

## Outputs

| File | Content |
|---|---|
| `<run>/fn_buckets/fn_services.csv` | One row per FN service: cell keys, score, training history of its cell, CPT family, bucket |
| `<run>/fn_buckets/bucket_summary.csv` | Per bucket: FN services, share, median score, median training PA rate and size, top CPTs |
| `<run>/fn_buckets/run_summary.json` | Cut-off, TP / FP / FN / TN, precision, all settings used |
| `fn_bucket_history.csv` (working folder) | One row per run: FN per bucket. Compare only rows with the same `test` column |
| `fn_bucket_cache/` | Training history per data setting, so later runs on the same data don't reload 5 years |

## How it works

1. **Training history per cell** – reads the std claims files one year at a time and prepares them exactly as the training pipeline does: label TRUE/FALSE only (AMBIGUOUS dropped), quality codes kept / set FALSE / excluded per the run, secondary claims kept, repeated lines (CSN + CPT + modifier + date + filing order) removed keeping the TRUE copy, payer and plan names cleaned. It keeps lines before the test window and counts lines and PA rate per cell. Checked against the pipeline: identical cell counts and rates.
2. **Test services** – CSN + CPT; PA if any line is PA; score = highest line score; cell = modifier, place of service and payer | plan of the highest-scoring line.
3. **Cut-off and FN** – the score that catches 80% of PA services on the test set; FN = PA service scored below it.
4. **Buckets** – first rule that matches wins.

| Bucket | Rule | Source |
|---|---|---|
| 1 Likely label noise | Quality code, or an E/M or lab service on a visit that also has a PA "anchor" service | **Assumption** (heuristic by CPT family; does not check auth numbers) |
| 2 Unseen or rare cell | Fewer than 20 training lines for the exact cell | Threshold chosen by us |
| 3 Historically non-PA | Training PA rate below 5% | Threshold chosen by us |
| 4 Mixed cell | Training PA rate 5–50% | Thresholds chosen by us |
| 5 Model miss | Training PA rate at least 50%, yet scored below the cut-off | Data |

A **cell** is CPT + modifier + place of service + payer | plan, the model's inputs.

## What comes from our own knowledge, not from the data

These are choices to review, not facts the data proved:

1. **CPT families** (in `FAMILIES`, top of the script) come from the general CPT / HCPCS code structure and should be checked by the coding team:

   | Family | Codes | Meaning |
   |---|---|---|
   | quality | `####F` | CPT Category II quality-reporting codes ($0, never PA) |
   | E/M | 99200–99499, G2211 | Evaluation and management visits; G2211 visit-complexity add-on |
   | lab | 80000–89999, 36415 | Pathology and laboratory; 36415 = blood draw |
   | anchor | 70000–79999, 10000–69999, J####, 97### | Radiology, surgery, J-code drugs, physical / rehab therapy |
   | other | everything else | |

   The ranges are broad: e.g. "surgery 10000–69999" includes minor procedures that rarely need PA, and 99200–99499 includes some non-visit E/M codes.
2. **Bucket 1 is a heuristic.** It assumes an E/M or lab service on a visit with a PA anchor is likely carrying the anchor's auth. It does not look at auth numbers. The auth-number evidence (the passenger check) is in `fn_root_cause.py` and the v8 passenger rule.
3. **Thresholds** – 20 lines for "rare", 5% and 50% for the rate bands, and 80% recall for the cut-off were chosen by the team for analysis, not set by the business. All are flags.
4. **The cut-off is read off the test set** (the same way as in the benchmark analysis). It is right for comparing runs, not a production cut-off.

## Reading the history

- Compare rows with the same `test` value only: same services, same answer key.
- A shrinking **bucket 1** means label noise is being removed. With the v8 labels it should be close to empty, because quality codes are excluded and passengers relabelled.
- **Buckets 3 and 4** shrinking need new information (pre-visit features). A smaller bucket 5 means the model ranks known PA cells better.
- With the default (80% recall) the total FN is fixed by design (20% of PA). Watch how the FN are **spread** across buckets, and the FP column, not the total.
- To see the total FN move, also run with `--cutoff 0.8`. The history keeps the two rules apart (`cutoff_rule` column). Remember that the same cut-off lands at a different recall for each model.
