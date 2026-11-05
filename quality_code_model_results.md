# PA Model – Quality Codes: Model Results for Three Training Modes

*5 Nov 2026 · Companion to the quality-code label analysis document · Comparison script: `compare_quality_modes.py`*

## Summary
- **Excluding quality codes (CPT Category II, `^\d{4}F$`) gives the best model on every view and cut-off.**
- On the same services with identical labels: AUC-PR 0.645 → **0.653**, FP at 80% recall 60,737 → **52,356 (−14%)**, FN at the best cut-off 15,547 → **14,617**.
- Against the reported benchmark: FN 18,070 → **14,617 (−19%)**. About 2,500 of that is quality-code "misses" that were never real PA; about 930 is genuine improvement.
- Labelling quality codes FALSE (keeping them as rows) is slightly **worse** than doing nothing.
- **Recommendation:** adopt *exclude* as the new base setting.

## Setup
Identical in all three runs; only the quality-code handling differs.

| Item | Value |
|---|---|
| Model | LightGBM `lgb_full`, 5 features: FINAL_CPT, FINAL_MODIFIERS, PLACE_OF_SERVICE_CODE, PAYOR_CLEAN, PAYOR_PLAN_CLEAN |
| Training period | 2022-01-01 .. 2026-04-30 (train + update windows) |
| Validation | Random 10% of visits (seed 42); early stopping on log loss (patience 50, max 2,000 trees); no refit |
| Filing order | No filter: secondary claims kept in train and test |
| Test | 2026-05-01 .. 2026-07-31 |
| Unit | Service = CSN + CPT (label = max of lines, score = max of lines); visit = CSN |

| Mode | Data / setting | Output folder | Fit + validation lines |
|---|---|---|---|
| **Keep** (benchmark) | v4, `GROUP_LABEL` | `exp_nofilter_2022` | 9,982,765 + 1,109,026 |
| **FALSE** | v6, `PA_LABEL_COL=LABEL_V6` | `exp_qc_false_rv` | 11,658,220 + 1,294,346 |
| **Exclude** | v4, `PA_QUALITY_MODE=exclude` | `exp_qc_exclude_rv` | 6,908,273 + 766,707 |

Commands (same flags; only the environment setting and output differ):
```bash
python -u pa_experiments.py baselines --train-years 2022 2023 2024 2025 --test-years 2026 --update-start 2026-02-01 --test-start 2026-05-01 --test-end 2026-08-01 --sample 1.0 --models lgb_full --include-secondary --secondary-in-test --out <folder>
python compare_quality_modes.py exp_nofilter_2022=keep exp_qc_false_rv=false exp_qc_exclude_rv=exclude --out qc_compare_rv
```

## How to read the three views
| View | Scope | Use |
|---|---|---|
| **A** | Non-quality services in each run's test set; labels identical across runs | **Main result: fair model comparison** |
| B | All services in each run's own test set | What each run reports by itself; includes the effect of removing quality "PA" |
| C | Visits, built from non-quality services | Visit labels barely change, so a gain here is real |

Cut-offs: **best_thr** = best F1 on a 0.05 grid (as in the experiment sheet); **@80% / @90%** = cut-off set per model so it catches 80% / 90% of PA (equal-recall comparison).

## A. Non-quality services (fair comparison)
| | Keep | FALSE | **Exclude** |
|---|---|---|---|
| Services / PA services | 1,501,135 / 49,751 | 1,500,679 / 49,739 | 1,502,265 / 49,781 |
| AUC-PR | 0.6447 | 0.6372 | **0.6527** |
| AUC-ROC | 0.9607 | 0.9601 | **0.9618** |
| TP / FP / FN @ best_thr 0.8 | 34,204 / 24,930 / 15,547 | 33,538 / 24,960 / 16,201 | 35,164 / 26,006 / **14,617** |
| FP @ 80% recall | 60,737 | 61,639 | **52,356** |
| Precision @ 80% | 39.6% | 39.2% | **43.2%** |
| FP @ 90% recall | 166,444 | 173,195 | **161,075** |
| Precision @ 90% | 21.2% | 20.5% | **21.8%** |

## B. Each run's own test set
| | Keep | FALSE | **Exclude** |
|---|---|---|---|
| Services / PA services | 1,949,656 / 52,278 | 2,157,541 / 49,739 | 1,502,265 / 49,781 |
| of which quality services / quality PA | 448,521 / 2,527 | 656,862 / 0 | 0 / 0 |
| AUC-PR | 0.6203 | 0.6370 | **0.6527** |
| TP / FP / FN @ best_thr 0.8 | 34,208 / 24,946 / **18,070** | 33,538 / 24,970 / 16,201 | 35,164 / 26,006 / **14,617** |
| FP @ 80% recall | 89,331 | 61,965 | **52,356** |
| Precision @ 80% | 31.9% | 39.1% | **43.2%** |

**Check:** the keep row reproduces the benchmark sheet exactly (AUC-PR 0.62029, best_thr 0.8, TP 34,208, FP 24,946, FN 18,070).

## C. Visit level (non-quality services)
| | Keep | FALSE | **Exclude** |
|---|---|---|---|
| Visits / PA visits | 819,033 / 23,404 | 818,762 / 23,397 | 819,665 / 23,417 |
| AUC-PR | 0.6723 | 0.6670 | **0.6765** |
| TP / FP / FN @ best cut-off | 13,890 / 6,933 / 9,514 (0.95) | 16,182 / 11,976 / 7,215 (0.90) | 14,188 / 7,256 / 9,229 (0.95) |
| FP @ 80% recall | 21,809 | 22,853 | **20,178** |
| Precision @ 80% | 46.2% | 45.0% | **48.2%** |

The visit best cut-off differs between runs (0.90 vs 0.95), so compare visit results at equal recall.

## Against the benchmark at the same best_thr (service level)
| | Benchmark (keep) | Exclude | Change |
|---|---|---|---|
| AUC-PR | 0.6203 | 0.6527 | +0.032 |
| TP | 34,208 | 35,164 | +956 |
| FP | 24,946 | 26,006 | +1,060 |
| FN | 18,070 | 14,617 | −3,453 (−19%) |
| Recall | 65.4% | 70.6% | +5.2 pts |
| Precision | 57.8% | 57.5% | ≈ same |
| F1 | 0.614 | 0.634 | +0.020 |

**Why FP rises slightly at the same cut-off:** removing ~3.9M negative training rows raises the training PA rate (≈4.0% → 5.5% of lines), so all scores shift up. The fixed 0.8 cut-off then sits at a higher recall (70.6% vs 65.4%). This is a cut-off effect, not worse ranking: **at equal recall, exclude has fewer FP** (table A).

## Interpretation
- **Exclude:** better ranking on real services (AUC-PR +0.008 on identical labels), with fewer FP at every matched recall. It also removes the ~2,500 quality-code "misses", which are label artefacts.
- **FALSE:** slightly worse than keep on real services. It adds ~2.2M easy negatives (quality lines that were AMBIGUOUS before), which appears to dilute training and validation. *Hypothesis, not tested.*
- **Keep:** the model is scored on 448k quality services, including 2,527 inherited PA labels it cannot predict.

## Caveats
- **One run per mode, random validation (seed 42).** The gains agree across all views and cut-offs; a second seed would confirm the size.
- **Non-quality test services differ by ~0.1% between runs** (1,500,679–1,502,265). Visits are assigned to windows by their first service date, and removing a quality line can move it. Too small to affect the conclusion.
- **Not yet run:** FP at the benchmark's own recall (65.4%), the cleanest single comparison with the sheet.
- **Reported figures change meaning:** test-set size, PA rate and FN counts differ by mode. Always state the mode with a number.

## Decision and next steps
- [ ] Adopt **exclude** as the base setting (`PA_QUALITY_MODE=exclude`, or `IS_QUALITY_CODE = 0` on v6)
- [ ] Add a rule: quality codes in a prediction request → "PA not required" (only if they can appear there; to confirm)
- [ ] Optional: second seed, and FP at 65.4% recall, before announcing final numbers
- [ ] Next experiment on top of exclude: passenger relabel (v6.1)
