# PA Labels – v8 Draft Dataset Analysis

Nov 10, 2026 · @Sid

## Summary

The v8 draft is the std v4 claims data (2022 to Jul 2026, 27.5M lines) with seven label columns side by side, so every label version can be trained and compared from one dataset. Nothing in v4 is dropped or overwritten.

- **Status:** draft. The passenger rule has not been tested in training or approved by the team. Location: `s3://cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v8_draft/`, with `v8_draft_manifest.json`.
- **Three label changes, built in layers:**
  - **v6, quality codes:** CPT Category II codes (`####F`) set FALSE (21,025 TRUE lines → FALSE).
  - **v7, Acuity auth-denial codes from Anesha's mail:** lines carrying one of the 17 codes set TRUE (+763,833 TRUE lines with all 17 codes; +4,369 with the 10 auth-only codes).
  - **v8 draft, passengers:** services that are TRUE only because another service's auth number is on the same claim set FALSE (23,531 lines in v6 labels).
- **No Acuity code ever turns a line FALSE.** The passenger rule keeps any line that carries its own Acuity code TRUE in the v7-based labels.
- **All counts are raw lines** (before the training pipeline drops AMBIGUOUS lines and repeats), unless stated otherwise.

| Label column | TRUE lines (2022–26) | vs original |
| --- | --- | --- |
| `GROUP_LABEL` (original) | 547,317 | — |
| `LABEL_V6` | 526,292 | −3.8% |
| `LABEL_V6_PAX` | 502,761 | −8.1% |
| `LABEL_V7_STRICT` | 530,661 | −3.0% |
| `LABEL_V7_STRICT_PAX` | 508,637 | −7.1% |
| `LABEL_V7` | 1,290,125 | +135.7% |
| `LABEL_V7_PAX` | 1,268,101 | +131.7% |

## Data and scope

All five yearly files from 2022 to 2026 were used for every step, and every row of v4 is kept in v8.

| Source | Path | Used for |
| --- | --- | --- |
| std v4 (one row per claim line) | `.../09282026/std_data_v4/ai_prior_auth_09282026_v3_{year}_std.parquet` | Base rows, `GROUP_LABEL`, auth number, CPT, payor |
| Raw remit v3 (one row per remit adjustment code) | `.../09282026/v3/ai_prior_auth_09282026_v3_{year}.parquet` | Acuity codes on **all** remit rows of a line (std keeps only the first remit row's codes) |
| Acuity code list | "Acuity – Auth Denials CARC Codes" (Anesha's mail) | The 17 auth-denial CARCs |

| Year | Lines (rows) | `GROUP_LABEL` TRUE |
| --- | --- | --- |
| 2026 (Jan–Jul) | 8,621,569 | 140,135 |
| 2025 | 10,099,438 | 205,537 |
| 2024 | 3,673,762 | 104,642 |
| 2023 | 2,638,365 | 47,538 |
| 2022 | 2,481,635 | 49,465 |
| **Total** | **27,514,769** | **547,317** |

Lines per year jump in 2025. Quality codes alone explain 3.6M of that jump (3.5% of lines in 2024, 37% in 2025), which points to a change in what the extract captures; not yet confirmed with the data team.

## Label columns and definitions

Each label column starts from the one before it and changes only the lines its rule names; values are TRUE, FALSE or AMBIGUOUS.

| Column | Built from | Rule | Direction of change |
| --- | --- | --- | --- |
| `GROUP_LABEL` | Production labelling script | Level 1 auth number field → Level 2 remit codes (15, 39, 197, 198, 210, 284, 296, 302, M62, N54) → collapse over remit rows → segment override | — (original) |
| `LABEL_V6` | `GROUP_LABEL` | Quality code (`FINAL_CPT` matches `^\d{4}F$`) → FALSE, whatever the original label | TRUE/AMBIGUOUS → FALSE |
| `LABEL_V7` | `LABEL_V6` | Any of the 17 Acuity codes on any remit row of the line → TRUE (quality codes excluded) | FALSE/AMBIGUOUS → TRUE |
| `LABEL_V7_STRICT` | `LABEL_V6` | One of the 10 auth-only Acuity codes on any remit row → TRUE (quality codes excluded) | FALSE/AMBIGUOUS → TRUE |
| `LABEL_V6_PAX` | `LABEL_V6` | Passenger → FALSE | TRUE → FALSE |
| `LABEL_V7_STRICT_PAX` | `LABEL_V7_STRICT` | Passenger → FALSE, unless the line has its own Acuity code | TRUE → FALSE |
| `LABEL_V7_PAX` | `LABEL_V7` | Passenger → FALSE, unless the line has its own Acuity code | TRUE → FALSE |

**Helper columns**

| Column | Meaning |
| --- | --- |
| `IS_QUALITY_CODE` | 1 if CPT Category II (`####F`); 84 distinct codes in 2022–26 |
| `ACUITY_AUTH_CODES` | Acuity codes found on the line's remit rows, e.g. `16,197` |
| `RELABEL_REASON` / `RELABEL_REASON_V7` | `quality_code_cpt2`, `acuity_auth_only`, `acuity_mixed` |
| `IS_PASSENGER` | 1 if the passenger rule applies |
| `VISIT_LEVEL_AUTH` | 1 if a listed code shares an auth but nothing else on the claim is TRUE; **kept TRUE** (referral question open) |
| `LABEL_VERSION` | `v8.0-draft` |

**Acuity code groups** (all have category Denial, sub-category Authorization / Pre-Certification in Anesha's sheet)

| Group | Codes | Why |
| --- | --- | --- |
| Auth-only (used by `LABEL_V7_STRICT`) | 39, 61, 62, 165, 197, 198, 210, 243, P17, W6 | The code's only row in the sheet is Authorization |
| Mixed (only in `LABEL_V7`) | 15, 16, 59, 96, 109, 147, A1 | The same code also maps to other sub-categories (billing error, coverage, coding…) in the sheet |

## Build steps, filters and groupings

The dataset was built in three layers, each a copy of the previous one plus new columns; no rows were filtered out at any layer.

**Layer 1 – v6, quality codes** (`build_v6.py`)

1. Read each v4 year in batches of 500k rows.
2. Flag `IS_QUALITY_CODE` where `FINAL_CPT` (trimmed, upper case) matches `^\d{4}F$`. Category III codes (`####T`) are not matched.
3. Set `LABEL_V6` = FALSE on those lines; all other lines copy `GROUP_LABEL`.
4. Check per year: v4 rows = v6 rows, and TRUE before − TRUE after = TRUE → FALSE.

**Layer 2 – v7, Acuity codes** (`build_v7.py`)

1. From the raw remit file, keep remit rows whose `CARC` column is one of the 17 codes. Remark codes such as M62 and N54 are also stored in this column, one code per remit row; the `RARC` column holds internal numeric IDs and was not used.
2. **Grouping:** collapse to one row per claim line on CSN + `FINAL_CPT` + `SERVICEDATE` (+ `FINAL_MODIFIERS` and `FILING_ORDER_C` where the raw file has them), listing the distinct codes found.
3. Join to v6 on the same key; lines with no match get no codes.
4. Set `LABEL_V7` / `LABEL_V7_STRICT` to TRUE where a qualifying code is present and the line is not already TRUE and is not a quality code.
5. Check per year: rows unchanged, and TRUE after − TRUE before = lines turned TRUE.

**Layer 3 – v8 draft, passengers** (`v6_label_impact.py` then `build_v7_1.py`)

1. **Auth value:** `LINE_AUTHORIZATION_NUMBER`, trimmed and upper case; blanks and "no auth" text (NO AUTH / NO REF / NOT REQ / NPN / NPAR / N/A / zeros…) treated as no auth.
2. **Grouping – claim:** CSN + `FILING_ORDER_C`. A line's auth is **shared** if the same auth value is on 2 or more distinct CPTs of the claim. A CPT is **alone** if it is the only distinct CPT on the claim.
3. **Grouping – per CPT (passenger list):** on all 2022–26 lines, quality codes excluded, original `GROUP_LABEL`, alone-on-claim PA rate = TRUE lines ÷ all alone lines (AMBIGUOUS counted as not PA). A CPT goes on the list if the rate is **< 2%** with **≥ 100** alone lines → **344 CPTs**.
4. **Grouping – per CPT + payor (exceptions):** for listed CPTs, alone rate per CPT + `PAYOR_CLEAN` (payor name upper-cased, "AS OF…" removed, spaces squeezed). Rate **≥ 5%** with **≥ 50** alone lines → exception, never relabelled → **123 pairs**.
5. **Passenger** = TRUE in `GROUP_LABEL` through an auth value, auth shared, CPT on the list, CPT + payor not an exception, not a quality code, **and** another TRUE non-quality, non-passenger line stays on the claim (the auth has an "owner").
6. Lines meeting all but the owner condition are kept TRUE and flagged `VISIT_LEVEL_AUTH`.
7. Write the three `_PAX` columns; check per year that rows are unchanged.

The list and exceptions were first derived on 2025–Apr 2026 (241 CPTs, 31 exceptions); they were rederived on all years to match training on all years. The all-years list flips 1,885 more lines, mostly labs (85025, 80053, 88305).

**Training-time steps (not applied in the dataset):** AMBIGUOUS lines dropped; repeated lines (CSN + CPT + modifier + date + filing order) removed keeping the TRUE copy; no filing-order filter; quality codes excluded (current base setting).

## Label changes

The full Acuity list multiplies PA lines by about 2.5, while the auth-only list and the passenger rule each move them by under 5%.

**Transitions by layer (raw lines, 2022–26)**

| Layer | TRUE → FALSE | FALSE → TRUE | AMBIGUOUS → TRUE | AMBIGUOUS → FALSE |
| --- | --- | --- | --- | --- |
| v6 quality codes | 21,025 | 0 | 0 | 2,547,289 |
| v7 full (17 codes) | 0 | 763,833 in total, split not yet computed | (included in total) | 0 |
| v7 strict (10 codes) | 0 | 4,369 in total, split not yet computed | (included in total) | 0 |
| v8 passengers on `LABEL_V6` | 23,531 | 0 | 0 | 0 |
| v8 passengers on `LABEL_V7` / `LABEL_V7_STRICT` | 22,024 | 0 | 0 | 0 |

The FALSE → TRUE versus AMBIGUOUS → TRUE split for v7 matters: the first flips an existing training example, the second adds a row the pipeline used to drop. It has not been computed yet (query under Open items).

**TRUE lines by year**

| Year | `GROUP_LABEL` | `LABEL_V6` | `LABEL_V6_PAX` | `LABEL_V7_STRICT` | `LABEL_V7_STRICT_PAX` | `LABEL_V7` | `LABEL_V7_PAX` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026 | 140,135 | 133,357 | 127,272 | 134,979 | 129,063 | 341,066 | 335,150 |
| 2025 | 205,537 | 192,217 | 184,663 | 193,207 | 186,042 | 436,515 | 429,350 |
| 2024 | 104,642 | 103,746 | 98,391 | 104,155 | 99,326 | 253,533 | 248,704 |
| 2023 | 47,538 | 47,507 | 45,364 | 48,225 | 46,278 | 131,116 | 129,169 |
| 2022 | 49,465 | 49,465 | 47,071 | 50,095 | 47,928 | 127,895 | 125,728 |
| **Total** | **547,317** | **526,292** | **502,761** | **530,661** | **508,637** | **1,290,125** | **1,268,101** |

**v7: lines turned TRUE, by Acuity code** (a line can carry more than one code)

| Code | Lines turned TRUE | Group |
| --- | --- | --- |
| 59 Multiple-procedure rules | 373,743 | Mixed |
| 16 Lacks information / billing error | 187,601 | Mixed |
| 96 Non-covered charge | 134,237 | Mixed |
| A1 Claim denied (generic) | 52,099 | Mixed |
| 109 Not covered by this payer | 26,071 | Mixed |
| 243 Not authorized by network/PCP | 2,729 | Auth-only |
| 197 Precert/auth absent | 1,297 | Auth-only |
| 147 Contracted rate not on file | 1,193 | Mixed |
| 165 Referral absent or exceeded | 294 | Auth-only |
| 39, 210, 198, 15, 61 together | 52 | — |

197 is already in the production rule, yet 1,297 lines carrying it were not TRUE in v6. A likely reason (not verified) is that the raw remit file holds remit rows added after the labels were built.

**v8 passengers by year**

| Year | Passengers set FALSE | Visit-level auth kept TRUE |
| --- | --- | --- |
| 2026 | 6,085 | 3,075 |
| 2025 | 7,554 | 1,945 |
| 2024 | 5,355 | 820 |
| 2023 | 2,143 | 655 |
| 2022 | 2,394 | 625 |
| **Total** | **23,531** | **7,120** |

Most frequent passenger CPTs: 99214, 99213, G2211, 36415 (blood draw), 93000 (ECG), 85025 (blood count), 80053 (metabolic panel), 88305 (pathology), 99215, 99204 and eye exams (92015, 92060, 92134).

**After the training pipeline** (AMBIGUOUS dropped, repeats removed) only v6 has been counted so far: 515,956 PA lines, 475,513 PA services, 252,720 PA visits (2022–26). The v7 and v8 counts will come from the training logs.

## Did any lines with Anesha's codes end up FALSE?

No Acuity code ever sets a line to FALSE, but in two specific places lines that carry an Acuity code are FALSE.

| Where | Lines | What happens | Why |
| --- | --- | --- | --- |
| Quality codes in `LABEL_V7` / `LABEL_V7_STRICT` | 38,205 carry an Acuity code (77 of them were TRUE in `GROUP_LABEL`) | Kept FALSE | Quality codes are $0 reporting tags with no PA; agreed to keep them FALSE for now |
| Passengers in `LABEL_V6_PAX` | Up to 1,507 | Set FALSE | `LABEL_V6_PAX` is built on v6 and does not look at Acuity codes |
| Passengers in `LABEL_V7_PAX` / `LABEL_V7_STRICT_PAX` | 1,507 | **Kept TRUE** | Lines with their own Acuity code are exempt from the passenger rule (23,531 passengers − 22,024 flipped) |

**Quality-code lines kept FALSE, by Acuity code** (2022–26)

| Code | Lines | Group |
| --- | --- | --- |
| 16 Lacks information / billing error | 33,172 | Mixed |
| 96 Non-covered charge | 4,485 | Mixed |
| 109 Not covered by this payer | 855 | Mixed |
| A1 Claim denied | 504 | Mixed |
| 197 Precert/auth absent | 37 | Auth-only |
| 39 Denied at auth request | 24 | Auth-only |
| 59, 243 | 8 | — |

Original `GROUP_LABEL` of those 38,205 lines: AMBIGUOUS 32,965, FALSE 5,163, TRUE 77. Making quality codes follow the strict list would change at most 63 lines.

**One inconsistency to fix:** the exemption in `LABEL_V7_STRICT_PAX` checks for any of the 17 codes, not only the 10 auth-only ones. A passenger whose only Acuity code is a mixed code (e.g. 59) therefore stays TRUE in the strict label, even though the strict label does not treat 59 as auth. The number of such lines is at most 1,507 and has not been counted.

## External check against X12 and CAQH

The official code definitions support the strict list: every auth-only code is explicitly about authorization or referral, and none of the six mixed codes mentions authorization.

| Code | Official X12 description | Status | Verdict |
| --- | --- | --- | --- |
| 39 | Services denied at the time authorization/pre-certification was requested | Active | Auth |
| 61 | Adjusted for failure to obtain second surgical opinion | Active | Auth-type |
| 197 | Precertification/authorization/notification/pre-treatment absent | Active | Auth (includes "notification") |
| 198 | Precertification/notification/authorization/pre-treatment exceeded | Active | Auth |
| 210 | Pre-certification/authorization not received in a timely fashion | Active | Auth |
| 243 | Services not authorized by network/primary care providers | Active (replaced 38) | Referral / network |
| P17 | Referral not authorized by attending physician; property & casualty only | Active | Not relevant to health claims |
| W6 | Replaced by P17 | Deactivated | Cannot occur (0 lines) |
| 62, 165 | Not on the current lists checked | Probably deactivated (not verified) | 62: 0 lines; 165: 294 lines, to check |
| 15 | Authorization number missing, invalid, or does not apply to the billed services or provider | Active | Auth |
| 284 / 296 / 302 | Auth number valid but not for these services / this provider; auth time limit expired | Active (2017 / 2018 / 2020) | Auth; **not in Acuity's list**, already in our rule |
| 16 | Claim/service lacks information or has submission/billing error(s) | Active | Not auth |
| 59 | Processed based on multiple or concurrent procedure rules | Active | Not auth |
| 96 | Non-covered charge(s) | Active | Not auth |
| 109 | Claim/service not covered by this payer/contractor | Active | Not auth |
| 147 | Provider contracted/negotiated rate expired or not on file | Active | Not auth |
| A1 | Claim/service denied; use only when no more specific code is available | Active | Generic |

**CAQH CORE** (the operating-rules body for remittances) defines a denial's meaning by the **combination** of group code (CO/PR/OA/PI), CARC and optional RARC, not by the CARC alone. That is consistent with Acuity mapping 16, 59, 96, 109, 147 and A1 to Authorization only in some rows. Our raw file has no group-code column that we have found, and its RARC column holds internal IDs, so we cannot yet reproduce that combination mapping.

The official X12 page was not reachable from the analysis environment; descriptions come from a republished copy of the X12 list and a 2022 state code list.

Sources: [CARC list – claim.md](https://docs.claim.md/docs/claim-adjustment-reason-codes) · [CAQH CORE CARC/RARC rule](https://www.caqh.org/sites/default/files/core/Payment-Remittance-CARCs-RARCs-Rule.pdf) · [CT APCD CARC list (2022)](https://portal.ct.gov/-/media/ohs/health-it-advisory-council/apcd-advisory-group/data-submission-guide-workgroup/meeting-materials/6-30-22/carc-codes_final.pdf) · [X12 CARC list (official)](https://x12.org/codes/claim-adjustment-reason-codes)

## Caveats and open items

The dataset is complete and internally checked, but several numbers and decisions are still open before it can lose the draft tag.

**Caveats**

- All counts are raw lines; training counts will be lower after the AMBIGUOUS drop and removal of repeats.
- The 2% and 5% thresholds and the 100 / 50 line minimums are analyst choices, not derived values. Results at 1% and 2% were similar; 5% flips noticeably more.
- The passenger list includes a few procedure codes (e.g. 17110, 11900, 73620, 93970) that are less obviously passengers than office visits and labs; their flips are small but not individually reviewed.
- `LABEL_V7` changes the target itself (about 2.5× PA lines), so its model results cannot be compared with others on its own labels alone. Comparisons should also use the frozen test set (original labels) and visit level.

**Open items**

- [ ] Count v7's FALSE → TRUE versus AMBIGUOUS → TRUE split (query below)
- [ ] Decide whether the `LABEL_V7_STRICT_PAX` exemption should use only the 10 auth-only codes
- [ ] Check why 1,297 lines with code 197 were not TRUE in v6
- [ ] Check code 165 (probably deactivated, yet 294 lines) and whether referral codes 287 / 288 occur in our data
- [ ] Look for a group-code (CO/PR/OA/PI) column in the raw remit file
- [ ] Ask Anesha / Paula what decides the Authorization row for the mixed codes, and mention 284 / 296 / 302
- [ ] Team decision: do referrals count as PA (codes 165 / 243 and the 7,120 visit-level auth lines)?
- [ ] Team approval of the passenger rule; then train `LABEL_V6_PAX`, `LABEL_V7_STRICT_PAX`, `LABEL_V7_PAX` (two seeds, quality codes excluded) and compare on the frozen test set

**Query for the v7 transition split**

```python
import pandas as pd, pyarrow.dataset as ds, pyarrow.fs as pafs
fs = pafs.S3FileSystem()
P = "cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v8_draft/ai_prior_auth_09282026_v3_{y}_std.parquet"
U = lambda s: s.astype(str).str.strip().str.upper().map({"TRUE": "TRUE", "1": "TRUE", "FALSE": "FALSE", "0": "FALSE"}).fillna("AMBIGUOUS")
out = []
for y in [2022, 2023, 2024, 2025, 2026]:
    for b in ds.dataset(P.format(y=y), filesystem=fs).to_batches(columns=["LABEL_V6", "LABEL_V7", "LABEL_V7_STRICT"], batch_size=1_000_000):
        t = b.to_pandas().apply(U)
        out.append(t.value_counts().rename("lines").reset_index().assign(year=y))
M = pd.concat(out).groupby(["LABEL_V6", "LABEL_V7", "LABEL_V7_STRICT"]).lines.sum()
print(M.to_string())
```
