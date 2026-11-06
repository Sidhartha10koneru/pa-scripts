# PA Model – Quality Codes: Model Results for Three Training Modes

*Updated 6 Nov 2026 · Companion to the quality-code label analysis document · Scripts: `compare_quality_modes.py`, `bootstrap_compare.py`, `register_run.py`*

## Summary
- **Excluding quality codes (CPT Category II, `^\d{4}F$`) gives a small but consistent real improvement on every measure.**
  - On the official test set (non-quality services, identical labels for all models), AUC-PR goes from 0.6447 to **0.6527 (+0.008)**, and the gain is the same under two training seeds.
  - At 80% recall there are **6–14% fewer false alarms** (−3.3k to −8.4k), and at 90% recall 3–4% fewer.
  - At the same false-alarm level, the model catches **+271 to +747 more PA (+0.7% to +1.9%)**.
- The headline FN drop against the benchmark (18,070 → 14,617) is **mostly a definition change**: about 2.5k quality-code "PA" services are no longer counted. Only about 700–900 fewer misses come from the model itself.
- Labelling quality codes FALSE (keeping them as rows) is slightly **worse** than doing nothing.
- **Recommendation:** adopt *exclude* as the new base setting, mainly for correctness, with a small performance gain. Bigger gains need the passenger label fix and new pre-visit features.

## Official test set (frozen: `test_v1_2026-05_2026-07`)
Built from the benchmark run by `freeze_test_set.py`. It is stored with a manifest and sha256 checksums, and every run is scored on exactly these services.

| Scope | Lines | Visits | Services | PA lines | PA services | PA visits | PA service rate |
|---|---|---|---|---|---|---|---|
| All (as reported so far) | 2,182,493 | 835,591 | 1,949,656 | 58,421 | 52,278 | 23,415 | 2.68% |
| **Official (quality codes excluded)** | **1,716,204** | **819,033** | **1,501,135** | **55,575** | **49,751** | **23,404** | **3.31%** |

**How the test set is defined**
- **Data:** std v4, 2022–2026 loaded.
- **Labels:** GROUP_LABEL TRUE/FALSE; AMBIGUOUS dropped.
- **Filing order:** no filter.
- **Duplicates:** repeated CSN + CPT + modifier + date + filing-order lines are removed, keeping the TRUE copy.
- **Test window:** lines dated 2026-05-01 to 2026-07-31, where the visit's first service date (taken from primary rows) is also in that window.
- **Service:** CSN + CPT; it is PA if any of its lines is PA.

The test set is reproduced exactly by `derive_test_set.py --years 2025 2026`. With 2026 alone there are 153 extra visits (522 services, 0 PA). Their earlier lines are dated Mar–Dec 2025 (verified).

## Setup
Identical in all runs; only the quality-code handling and the seed differ.

| Item | Value |
|---|---|
| Model | LightGBM `lgb_full`, 5 features: FINAL_CPT, FINAL_MODIFIERS, PLACE_OF_SERVICE_CODE, PAYOR_CLEAN, PAYOR_PLAN_CLEAN |
| Training period | 2022-01-01 .. 2026-04-30 |
| Validation | Random 10% of visits (`rv10`); early stopping on log loss (patience 50, max 2,000 trees); no refit |
| Filing order | No filter: secondary claims kept in train and test |
| Test | Frozen official set above |

## Registry (official test set, `registry.csv`)
| Run ID | Status | Fit / validation lines | Coverage | AUC-PR | FP @ 80% | Precision @ 80% | FN @ best_thr 0.8 | Visit AUC-PR |
|---|---|---|---|---|---|---|---|---|
| pa-lgb5-v4-grp-qckeep-rv10-s42-y22-26-t2605-2607-20261030 | benchmark | 9,982,765 / 1,109,026 | 1.0 | 0.6447 | 60,737 | 39.6% | 15,547 | 0.6723 |
| pa-lgb5-v4-grp-qcexcl-rv10-s42-y22-26-t2605-2607-20261104 | candidate | 6,908,273 / 766,707 | 1.0 | **0.6527** | **52,306** | **43.2%** | 14,607 | 0.6765 |
| pa-lgb5-v6-l6-qcfalse-rv10-s42-y22-26-t2605-2607-20261104 | archived | 11,658,220 / 1,294,346 | 0.9997 | ≈0.637 | — | — | 16,201 | 0.6670 |
| pa-lgb5-v4-grp-qckeep-rv10-s7-y22-26-t2605-2607-20261105 | archived (seed check) | 9,981,400 / 1,110,391 | 1.0 | 0.6445 | 55,243 | 41.9% | 15,665 | 0.6724 |
| pa-lgb5-v4-grp-qcexcl-rv10-s7-y22-26-t2605-2607-20261105 | archived (seed check) | 6,907,778 / 767,202 | 1.0 | **0.6527** | **51,920** | **43.4%** | 15,000 | 0.6772 |

**Notes on the registry**
- **Training and validation counts are not fixed numbers.** They depend on the quality-code mode and the seed. The definition is fixed, and the exact counts are recorded in each run card.
- **The FALSE run is missing 456 frozen services.** *Likely* cause (not verified): quality lines that were AMBIGUOUS became FALSE and were kept, which can move a visit's first date out of the test window.
- **The FALSE run was first registered with a wrong ID** (`v4-grp`), because it predates provenance recording. It should be re-registered under the v6/l6 ID shown above.

## Is the gain real, beyond dropping quality-code FN?
**1. Same services, same labels, two training seeds**

| Metric | Seed 42: keep → exclude | Seed 7: keep → exclude |
|---|---|---|
| AUC-PR | 0.6447 → 0.6527 (+0.0080) | 0.6445 → 0.6526 (+0.0081) |
| AUC-ROC | 0.9607 → 0.9618 | 0.9611 → 0.9620 |
| FP @ 80% recall | 60,737 → 52,356 (−13.8%) | 55,243 → 51,970 (−5.9%) |
| Precision @ 80% | 39.6% → 43.2% | 41.9% → 43.4% |
| FP @ 90% recall | 166,444 → 161,075 (−3.2%) | 167,158 → 160,373 (−4.1%) |
| Visit FP @ 80% recall | 21,809 → 20,178 | 21,702 → 19,798 |

These figures come from the comparison script, on each run's own non-quality services. The frozen-set figures in the registry differ by fewer than 100 FP.

The AUC-PR gain is identical across the two seeds. The FP gain at 80% recall varies with the seed, because the base model's own FP moves by about 5k between seeds. **Lesson:** a single-seed FP difference smaller than about 5k is within noise. Judge future experiments on AUC-PR, and use two seeds before quoting FP gains.

**2. Test-sample bootstrap** (seed-42 pair; 200 paired visit resamples; 1,501,135 common services; 0 label mismatches)

| Metric | Base | Exclude | Difference | 95% interval | Exclude better in |
|---|---|---|---|---|---|
| AUC-PR | 0.6447 | 0.6527 | +0.0081 | +0.0071 to +0.0093 | 200/200 |
| FP @ 80% | 60,736 | 52,297 | −8,439 (−13.9%) | −9,244 to −6,376 | 200/200 |
| Precision @ 80% | 39.6% | 43.2% | +3.6 pts | +2.9 to +3.9 | 200/200 |
| FP @ 90% | 165,708 | 160,925 | −4,783 (−2.9%) | −7,648 to −2,441 | 200/200 |
| Precision @ 90% | 21.3% | 21.8% | +0.5 pts | +0.3 to +0.8 | 200/200 |

The bootstrap covers only test-sample noise. Training randomness is covered by the seed rerun in table 1.

**3. TP at an equal false-alarm workload** (FP budget = the base model's FP at 80% recall)

| Seed | FP budget | TP base | TP exclude | Extra PA caught |
|---|---|---|---|---|
| 42 | 60,737 | 39,816 | 40,563 | +747 (+1.9%) |
| 7 | 55,243 | 39,807 | 40,078 | +271 (+0.7%) |

## All three modes, seed 42 (comparison script)
**A. Non-quality services (the fair comparison)**

| | Keep | FALSE | **Exclude** |
|---|---|---|---|
| AUC-PR | 0.6447 | 0.6372 | **0.6527** |
| FP @ 80% | 60,737 | 61,639 | **52,356** |
| Precision @ 80% | 39.6% | 39.2% | **43.2%** |
| FP @ 90% | 166,444 | 173,195 | **161,075** |
| TP / FP / FN @ best_thr 0.8 | 34,204 / 24,930 / 15,547 | 33,538 / 24,960 / 16,201 | 35,164 / 26,006 / 14,617 |

**B. Each run's own test set (what each run reports by itself)**

| | Keep | FALSE | Exclude |
|---|---|---|---|
| Services / PA | 1,949,656 / 52,278 | 2,157,541 / 49,739 | 1,502,265 / 49,781 |
| AUC-PR | 0.6203 | 0.6370 | 0.6527 |
| TP / FP / FN @ 0.8 | 34,208 / 24,946 / **18,070** | 33,538 / 24,970 / 16,201 | 35,164 / 26,006 / **14,617** |
| FP @ 80% | 89,331 | 61,965 | 52,356 |

The keep column reproduces the benchmark sheet exactly (AUC-PR 0.62029, FN 18,070).

**Why FP rises slightly at the fixed 0.8 cut-off:** excluding quality codes raises the training PA rate (≈4.0% → 5.5% of lines), so all scores shift up and 0.8 sits at a higher recall (70.6% vs 65.4%). This is a cut-off effect, not worse ranking: at equal recall, the excluded model has fewer FP.

## Production
1. Any quality code (`####F`) in a prediction request is answered "PA not required" by rule. Whether quality codes ever arrive in a request is still to be confirmed with the integration owner.
2. Every other service is scored by the model trained with quality codes excluded.
3. The cut-off is re-set on recent validation data at each retrain, for the agreed recall. Scores shift between trainings, so a cut-off is never carried over.

## Caveats
- Two seeds and one bootstrap. The direction is consistent; the size of the FP gain is a range, not a point estimate.
- Line, service and visit PA rates, test-set size and FN counts all depend on the quality-code mode. **Always state the mode and the test set with any number.**

## Next steps
- [ ] Team sign-off: exclude becomes the benchmark (`register_run.py … --status benchmark`)
- [ ] Re-register the FALSE run with its v6/l6 ID
- [ ] Agree the S3 location, then push the registry and the frozen test set
- [ ] Next experiment, on top of exclude: passenger relabel (v6.1), two seeds, registered
