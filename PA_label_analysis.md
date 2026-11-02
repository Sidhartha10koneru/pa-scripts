# PA Model – Label Analysis

*Dataset: `std_data_v4` (`ai_prior_auth_09282026_v3_{year}_std.parquet`) and raw remit rows (`v3/ai_prior_auth_09282026_v3_{year}.parquet`). Status as of 2 Nov 2026.*

## Summary

- **TRUE labels are genuine.** Every TRUE line checked is backed by an auth number or a real PA denial code.
- **The main problem is which lines are dropped, not wrong TRUE labels.** Any remit code that is not a PA denial makes a line AMBIGUOUS, and AMBIGUOUS lines are dropped. In groups that contain any PA, most normal paid lines are dropped, so PA rates look about **3× too high** in 2025 training data.
- **The inflation differs by year** (68% of lines dropped in PA groups in 2025, 35% in 2026), so training and test labels are built inconsistently.
- **Secondary issue:** one auth number on a claim makes every line on it TRUE, so quality codes and some office visits are labelled PA as passengers.
- **This explains** the strange 99214 PA rates (65–96%), the model's overprediction on some plans, poor calibration (scores about 4× too high) and much of the apparent 2025 → 2026 drift.

## Why we looked at labels

- The model is near its ceiling (visit AP 0.736 vs leave-one-out ceiling 0.745), so label quality matters more than model changes.
- 99214 (routine office visit) showed 65–96% PA for some plans in 2025: UHC Community Plan 96%, Anthem Commercial 65%.
- The model overpredicted those plans in 2026 test data (mean scores 0.36–0.63 vs 1–5% actual).

## How a label is made (labelling script)

| Step | Input | Result |
|---|---|---|
| Level 1 | Auth number field | Real auth number → TRUE; "no auth" text → FALSE; else UNCONFIRMED |
| Level 2 (UNCONFIRMED only) | Remit codes (CARC/RARC) | PA denial (15, 39, 197, 198, 210, 284, 296, 302, M62, N54) → TRUE; no codes at all → FALSE; **any other code → AMBIGUOUS** |
| Collapse | All remit rows of a line | Any TRUE → TRUE; else any FALSE → FALSE |
| Segment override | CPT + payor + plan group (per year file) | If the group has **zero** TRUE and >20 lines: AMBIGUOUS → FALSE |
| Model prep | — | AMBIGUOUS lines dropped |

## Findings

### 1. TRUE labels are evidence-backed ✅
- 99214 TRUE lines (572 strictly alone): 250 with an auth number, 247 with a PA denial code, 75 with a denial on another remit row.
- Genuine PA denials across 2025 remit rows: ~29,300 rows, 91% code 197 (auth absent).
- Code matching in the production script uses exact `isin`, so remark codes like MA15 are **not** read as 15. (An early audit of mine had that bug; corrected.)
- M62/N54 matching: no false matches found (1,513 rows, all in the CARC column).

### 2. AMBIGUOUS lines are mostly normal paid lines ❌ (main issue)
- 87% of raw remit rows are AMBIGUOUS at Level 2.
- Top codes: 45 contractual adjustment (5.7M), 3/2/1 copay/coinsurance/deductible (1.57M / 0.79M / 0.29M), MA01 appeal notice (1.2M), 253 sequestration (0.94M), MA15 claim split (0.94M), MA18 forwarded to secondary (0.88M), 23 prior payer (0.47M).
- These codes mean "processed and paid normally", which is evidence for FALSE when no auth exists.
- Final 2025 std file: **AMBIGUOUS 51%**, FALSE 47%, TRUE 2%.

### 3. Dropping AMBIGUOUS inflates PA rates where PA exists ❌

| CPT + payor + plan groups | Year | AMBIGUOUS dropped | PA rate as labelled | PA rate if AMBIGUOUS = FALSE | Inflation |
|---|---|---|---|---|---|
| With some PA | 2025 | **68.2%** | 9.0% | 2.9% | **3.1×** |
| With some PA | 2026 | 35.3% | 6.2% | 4.0% | 1.5× |
| With no PA | 2025 | 8.9% | 0% | 0% | — |

The segment override only rescues groups with zero PA, so in any group with even one PA line, the normal paid lines are dropped.

### 4. 99214 example

| Plan (2025) | TRUE | FALSE | AMBIGUOUS | As labelled | If AMBIGUOUS = FALSE |
|---|---|---|---|---|---|
| UHC Community Plan | 1,899 | 89 | 10,068 | 95.5% | 15.8% |
| Anthem HealthPlus | 275 | 126 | 10,186 | 68.6% | 2.6% |
| Anthem Commercial | 273 | 145 | 11,815 | 65.3% | 2.2% |
| Aetna Medicare Replacement | 129 | 42 | 14,758 | 75.4% | 0.9% |
| MVP | 301 | 437 | 14,540 | 40.8% | 2.0% |
| Fidelis Medicaid | 175 | 304 | 26,787 | 36.5% | 0.6% |

For most plans, 99214 is realistically 1–3% PA. Genuinely high and worth an SME question: UHC Community Plan (~16%), Fidelis Medicare (~40%, only 193 lines). Many 99214 auths are reused across visits (62%), which suggests referral-type authorisations.

### 5. Shared auth numbers create passenger TRUE labels ⚠️
One auth number on a claim makes every line TRUE (Level 1). Lines sharing another line's auth, by group:

| Group | Examples (lines) | Verdict |
|---|---|---|
| Quality codes (CPT II "F" codes) | 3074F 15.8k, 3008F 10.9k, 3078F 8.6k, 3079F 4.7k, 1159F 4.1k ≈ **47.5k** | **Noise**: reporting codes never need PA |
| Office visits | 99214 8.2k, 99213 5.7k, 99204 2.3k ≈ **16k** | **Probably noise** at line level |
| Parts of an authorised service | Contrast (A9575, Q9967, A9579), chemo admin (96413, 96375), PT (97140, 97110), nuclear stress (78452) | **Mostly legitimate**: one auth covers the whole study or treatment |

These matter most for service-level (CSN + CPT) evaluation; at visit level the visit is PA either way.

### 6. Secondary claims are not a label problem ✅
- 2026: 686,808 labelled secondary lines; 75% have a primary twin (same CSN + CPT + modifier + date); 97.8% go to a different payor.
- Labels agree with the primary twin in **99.96%** of cases; **0** PA secondary lines lack a primary twin.

## What this explains in the model
- Overprediction on 99214 for Anthem and Aetna plans (learned inflated 2025 rates).
- Poor calibration: mean visit score ~10% vs 2.8% actual.
- Apparent drift: Anthem Commercial 99214 "65% in 2025 → 5.5% in 2026" is mostly the artefact (corrected 2025 rate 2.2%).
- 2022–2024 training data hurts the model: those years also drop 51–54% as AMBIGUOUS and show 4–6% PA vs 2.9% in 2026.
- **Current metrics are measured on labels with the same bias**, so they will move after a fix.

## Proposed fixes (need SME sign-off before the dataset is finalised)
1. **"Paid normally" code list → FALSE.** When Level 1 is UNCONFIRMED and the line has only paid-normally codes (45, 1, 2, 3, 23, 253, MA01, MA15, MA18 …), label FALSE instead of AMBIGUOUS. Apply the same rule to every year.
2. **Shared auth on a claim:** TRUE only for the anchor service or CPT families that can require PA; quality F codes always FALSE.
3. **SME questions:** do referral auths count as PA (99214 at UHC Community Plan)? Is contrast/drug on an authorised study its own PA service? Which remaining codes (e.g. M51) stay AMBIGUOUS?
4. **Then:** rebuild labels, retrain, and compare old vs new labels on the same test window.

## Open checks
- **Safety of fix 1:** confirm known-PA services (MRI 70553/72148, PET 78815, Botox 64615) keep a high PA rate when AMBIGUOUS = FALSE. If they collapse, auth numbers are missing on paid lines and the rule must be restricted to CPT families that rarely need PA.
- **Which codes make up the dropped lines in PA groups** (std file CARC for AMBIGUOUS lines in groups with PA) to finalise the paid-normally list.
- Refine the passenger audit with the anchor-service logic (initial version used an own-evidence rate per CPT computed before the audit bug fix).

## Pipeline fix already made
Repeat removal (same CSN + CPT + modifier + date + filing order) now keeps the TRUE copy when copies disagree. The old rule kept the first copy and lost 337 PA visits in the May–Jul 2026 test set.
