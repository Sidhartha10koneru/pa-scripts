# Payor Master Sheet – Design, Method and Validation Review

**Project:** Pre-authorisation (PA) prediction – linking the PRD rules file to 2026 claims
**Author:** Sid
**Date:** 14 October 2026
**Status:** Master sheet built and validated (v5). Label fix and model build are next.
**Scripts:** `run_mapping.py`, `validate_mapping.py`, `payor_tagging.py` (this repo)

---

## 1. Summary

| | Result |
|---|---|
| Claims in scope (2026 std file) | 8.62M lines, 2,437 distinct `PAYOR` + `PLAN` pairs |
| **Linked to a rules payor** | **83.5%** (7.20M claims) |
| Set aside for a known reason | 14.2% (supplement, BlueCard, federal, delegated, not in rules) |
| Unmatched (long tail) | 2.3% |
| Validation, where a coded list exists | **78% of volume fits strongly**, 0.2% does not fit |
| Medicare external check | **51 / 51** CMS published OPD PA codes present in `CMSHSP` |
| Linked volume with **no coded list** | **45.6%**, which is the model's target segment (per PRD) |

**In one line:** the master sheet translates claims payor names into rules payor codes. That lets each claim be checked against *its own* payor's PA list, and it splits the data into a rules segment (coded lists) and a model segment (non-coded lists).

---

## 2. Why a master sheet is needed

### 2.1 The two sources name payors differently

| Claims `PAYOR` / `PLAN` | Rules `PAYOR_CODE` / `PAYOR_NAME` |
|---|---|
| MEDICARE / MEDICARE PART A AND B | `CMSHSP` / CMS Medicare |
| EMPIRE PLAN / EMPIRE PLAN | `NYSHIP` / New York State Health Insurance Program |
| EMBLEM HEALTH / GHI / NYCE PPO PLAN | `GHINY` / EmblemHealth (GHI) |
| ANTHEM BCBS / ANTHEM BCBS COMMERCIAL | `EMHECH` / Anthem Blue Cross Blue Shield of New York |
| LOCAL 1199 / LOCAL 1199 | `1199NB` / 1199SEIU Family of Funds |

Exact-name matching linked only about 20 of 585 payor names.

### 2.2 CPT alone is not enough
Before payor matching, we checked each claim's CPT against **any** rules list:

| Label | CPT on any rules list |
|---|---|
| TRUE (PA) lines | 85% |
| FALSE (no PA) lines | 52% |

There are 139 payors with different lists, so almost any CPT is on *someone's* list. The useful question is whether the CPT is on **this claim's payor's** list, and answering it needs the payor mapping.

---

## 3. Inputs

| Input | Description | Preparation |
|---|---|---|
| Rules file (PRD, sheet "Plans with Services") | Payor/plan PA policies, about 440k rows, 139 payors | Multi-code CPT field exploded into one row per CPT. Codes glued together by Excel line breaks (`_x000D_`) split apart. Result: `exploded_rules_v2` (457k rows, 18.4k distinct codes) |
| Coded-list filter | PRD: rules are reliable only for `CODING_STATUS = CODED_LIST` | Only coded-list CPTs are used for checking |
| Claims (`ai_prior_auth_..._v3_2026_std.parquet`) | 8.62M lines; `PAYOR`, `PLAN`, `FINAL_CPT`, `GROUP_LABEL`, `PAT_ZIP`, `CSN` | Used as-is |

**`GROUP_LABEL`** (built upstream): TRUE when a line authorisation number matches the regex, or when one of 7 CARC/RARC auth-denial codes is present. FALSE when there is no auth evidence. Everything else is AMBIGUOUS. Validation uses TRUE/FALSE only.

---

## 4. How the mapping works

### 4.1 The principle
The mapping is **not fuzzy matching**. For each high-volume claims name we identified **the organisation behind it**, then linked it to the rules payor that is **that same organisation**. The decisions are recorded as explicit rules in `run_mapping.py`. The script applies them; it does not guess.

### 4.2 How a rule is written and applied
Each rule reads: *if the text `PAYOR | PLAN` (upper case) matches pattern X, assign rules code Y.*

```python
(r"^MEDICARE \|",   "CMSHSP"),   # payor field is exactly "MEDICARE"
(r"EMPIRE PLAN",    "NYSHIP"),
(r"OXFORD",         "OXFORD"),
(r"\bAARP\b",       "UHC"),
```

- Rules are checked **top to bottom, and the first match wins**. Specific brands come before their parent company: Oxford is checked before UnitedHealthcare because Oxford has its own PA list.
- **Overrides** are applied last and always win. They cover cases where the plan name says something the payor name hides, such as a supplement plan under a "UNITED HEALTHCARE" payor.
- About 40 rules cover all 2,437 pairs, because many names share a company word.

### 4.3 How each rules code was chosen
1. **The script proposed candidates.** `make_token_sheet` split claims payor names into words, ranked them by claim volume, and searched the rules `PAYOR_NAME` column for each word.
2. **A person chose the code.** Example: "MEDICARE" matched several rules payors (CMS Medicare, Anthem Medicare Advantage, HealthSpring Medicare Advantage…). All except `CMSHSP` are private insurers, and a claim whose payor is just "MEDICARE" is traditional Medicare, run by CMS. So the choice was `CMSHSP`.
3. **The choice was written as a rule**, so it applies to every matching claim.
4. **It was then validated** (section 6).

### 4.4 Four types of mapping

**Type A – Same organisation, same name**

| Claims PAYOR | Pattern | Rules code |
|---|---|---|
| OXFORD HEALTH PLANS | OXFORD | `OXFORD` |
| AETNA | AETNA | `AETNA` |
| CIGNA | CIGNA | `CIGNA` |
| MVP HEALTH CARE | MVP | `NYMVP` |
| LOCAL 1199 | 1199 | `1199NB` |
| HEALTHFIRST MEDICAID | HEALTHFIRST | `NYHFST` |
| METROPLUS HEALTH MEDICAID | METROPLUS | `METPL` |
| HUMANA MEDICARE | HUMANA | `HUMANU` |
| PACIFICSOURCE HEALTH PLAN | PACIFICSOURCE | `PSHLTH` |

**Type B – Same organisation, different name** (domain knowledge, documented reason)

| Claims PAYOR / PLAN | Rules code | Reason |
|---|---|---|
| MEDICARE / PART A AND B, PART B NY NGS, PART B OR, RAILROAD | `CMSHSP` (CMS Medicare) | These are all traditional Medicare products. NGS is the CMS contractor for NY |
| EMPIRE PLAN | `NYSHIP` | The Empire Plan is NYSHIP's plan for NY state employees, **not** Empire BCBS |
| MEDICAID - NEW YORK | `NYCAID` | Same programme |
| BCBS OF OREGON / REGENCE BCBS | `ORRBCS` | Regence is the Blue plan for Oregon |
| EMBLEM HEALTH / GHI | `GHINY` | GHI is an EmblemHealth entity |
| EMBLEM HEALTH / HIP | `HIPNY` | HIP is an EmblemHealth entity |
| UNITED HEALTHCARE MEDICAID / COMMUNITY PLAN | `UHCNY` | UHC's NY Medicaid plan |
| AARP MEDICARE … | `UHC` | AARP Medicare plans are insured by UnitedHealthcare (confirmed by the data, section 6.3) |

**Type C – The name alone is ambiguous, so other data decides**

| Claims PAYOR / PLAN | Problem | Evidence | Rules code |
|---|---|---|---|
| ANTHEM BCBS / COMMERCIAL, BLUE ACCESS, PPO, HEALTHPLUS | Anthem has one code per state | Patient ZIP prefixes 105, 109, 115, 117, 125 = **New York** | `EMHECH` |
| ANTHEM BCBS / BCBS MEDICARE PPO, MED ADV DUAL, TURQUOISE, CENTENNIAL | Anthem name, but… | Patient ZIPs 870–871 = **New Mexico** | `NMBCBS` |
| ANTHEM BCBS / MEDIBLUE, MEDICARE | Medicare Advantage product | Plan name | `MEDADV` |
| BCBS OF CO / ANTHEM BCBS CO | State in the name | Plan name | `COBCBS` |
| KAISER … COLORADO | State in the name | Plan name | `COKAIS` |

**Type D – Set aside on purpose**

| Tag | Claims | Example | Reason |
|---|---|---|---|
| `SUPPLEMENT_NO_PA` | 386,222 | UNITED HEALTHCARE / AARP SUPPLEMENTAL | Medigap pays after Medicare; PA is handled by Medicare |
| `BLUECARD` | 319,578 | ANTHEM BCBS / BLUECARD, AHN ANTHEM BCBS INDIANA | Out-of-area Blue member; the home plan's rules apply |
| `FEP` | 98,670 | ANTHEM BCBS NY FEDERAL, BCBS FEDERAL CO | Federal Employee Program; national rules, not in the file. Note: the "NY FEDERAL" ZIPs are AZ/CO |
| `DELEGATED` | 6,028 | OPTUM CARE NETWORK OREGON AARP MEDICARE | Delegated medical group with its own auth rules |
| `NOT_IN_RULES` | 416,830 | Molina/Affinity NY, Northwell, Globe Life, UMR, Generic Commercial, One Call | Payor has no entry in the rules file (e.g. rules cover Molina in 11 states but not NY) |

### 4.5 Checks built into `run_mapping.py`
- Every code used must exist in `rules_df["PAYOR_CODE"]`. The script prints `All codes found in the rules file`.
- The result counts claims by final category (section 5).
- Output: `payor_master_review.xlsx`, one row per `PAYOR`+`PLAN` pair, ranked by volume, with `rules_payor`, `matched_pattern`, `matched_on`, `lob`, `state`, flags and a blank `corrected_code` column for review.

---

## 5. Coverage results (v5 run)

| Category | Claims | Share |
|---|---|---|
| **Linked to a rules payor** | **7,197,657** | **83.5%** |
| Not in rules | 416,830 | 4.8% |
| Supplement (no PA) | 386,222 | 4.5% |
| BlueCard | 319,578 | 3.7% |
| Unmatched | 196,584 | 2.3% |
| Federal (FEP) | 98,670 | 1.1% |
| Delegated | 6,028 | 0.1% |

The unmatched 2.3% is a long tail (Braven, SelectHealth, US Family Health Plan, Luminare and others, each 2–5k claims). Mapping it was judged low value.

---

## 6. Validation

### 6.1 Method (`validate_mapping.py`)
For each claims payor, using only TRUE/FALSE lines and excluding companion codes (section 7.1):

- **own_pa_cov:** share of the payor's PA lines whose CPT is on its mapped rules list
- **own_lift:** how many times more common listed CPTs are on PA lines than on non-PA lines
- **pa_claims_with_listed_line:** share of PA *claims* with at least one line on the list

| Status | Rule |
|---|---|
| FITS | lift ≥ 5 **and** PA coverage ≥ 25% |
| WEAK_FIT | lift 2–5 |
| DOES_NOT_FIT | lift < 2 |
| NO_CODED_LIST | the rules code has no coded CPT list, so it can't be tested |
| NOT_TESTABLE | fewer than 20 PA lines |

### 6.2 Results (5,738,648 linked TRUE/FALSE lines)

| Status | Share of lines |
|---|---|
| FITS | 41.0% |
| WEAK_FIT | 11.4% |
| DOES_NOT_FIT | **0.2%** |
| NO_CODED_LIST | 45.6% |
| NOT_TESTABLE | 1.8% |

Of the testable volume (FITS + WEAK + DOES_NOT_FIT), **78% fits** and **0.4% does not fit**.

**Strong fits (selected)**

| Claims payor | Rules code | Own lift | PA claims with listed line |
|---|---|---|---|
| Humana Medicare | `HUMANU` | 367 | 0.92 |
| Tricare For Life | `TRCRU` | 127 | 0.84 |
| Medicare | `CMSHSP` | 84 | 0.19 (only 48 PA claims) |
| Cigna | `CIGNA` | 41 | 0.69 |
| PacificSource | `PSHLTH` | 36 | 0.86 |
| Aetna | `AETNA` | 26 | 0.87 |
| UHC commercial | `UHC` | 5.1 | 0.58 |

**Weak fits and likely reasons**

| Claims payor | Rules code | Lift | Likely reason |
|---|---|---|---|
| UHC Medicare Advantage | `UHC` | 4.7 | Just below the threshold of 5 |
| Oxford | `OXFORD` | 3.8 | Some services go to a separate vendor list |
| NY Medicaid | `NYCAID` | 3.4 | Broad Medicaid rules |
| Aetna Meritain | `MERITN` | 7.8 | Strong separation but only 4% coverage |

**No coded list (largest):** Emblem GHI, Empire Plan (NYSHIP), Anthem NY (commercial and HealthPlus), MVP, Emblem HIP, 1199, Magnacare, Horizon, Moda, BCBS Oregon. This is a gap in the rules file, not a mapping error.

### 6.3 AARP – resolved with data
Two candidates: `AARP` and `UHC`. Scored on AARP's own PA vs non-PA lines:
- `UHC`: PA coverage 20.5%, non-PA 2.5%, **lift 8.8**
- `AARP`: coverage 14.4%, outside the top 10

**Decision:** AARP → `UHC`. The same test showed that Optum Care Network Oregon (about half of AARP's PA lines) fits much worse (13.7% vs 27.5%), so it is tagged `DELEGATED`.

### 6.4 Medicare – external confirmation
- **Source:** CMS, *List of HCPCS Codes for OPD Services Requiring Prior Authorization*
  https://www.cms.gov/files/document/opd-services-require-prior-authorization.pdf
  Programme page: https://www.cms.gov/data-research/monitoring-programs/medicare-fee-service-compliance-programs/prior-authorization-pre-claim-review-initiatives/prior-authorization-certain-hospital-outpatient-department-opd-services
- The list has 51 codes in 8 categories (blepharoplasty, botulinum toxin, panniculectomy, rhinoplasty, vein ablation, cervical fusion, spinal neurostimulators, facet joint interventions). Categories were added in 2020, 2021 and 2023; the list was last changed in August 2024 (64492 and 64495 removed).
- **Result: 51 of 51 codes are on `CMSHSP`.** `CMSHSP` has 581 codes, so it also includes other CMS programmes (to confirm with the rules team).
- **Claims data:** lift 84; the best realistic fit of the 139 payors.

### 6.5 Validation fixes applied (for transparency)
The first version of `validate_mapping.py` reported 97.7% "REVIEW". That came from two script errors, not mapping errors:
1. **Double counting.** The mapping table had one row per line of business, so the join multiplied claim lines (24.5M instead of 5.7M). Fixed by de-duplicating on `PAYOR`+`PLAN`.
2. **Wrong comparison.** Each payor was ranked against all 139 lists. Large lists from other states win that ranking simply by being large, because PA-type services are similar across insurers. Status is now based on the payor's **own** list. The "best other code" is kept for information only.

---

## 7. Findings beyond the mapping

### 7.1 The PA label is attached per claim, not per line
- 97% of TRUE **claims** contain at least one service on a rules list, so claim-level labels look reliable.
- But many TRUE **lines** are companion lines that inherit TRUE from the claim: office visits (99213/99214/99204), infusion administration (96365/96413/96415), blood draws (36415), contrast (Q9967/A9579) and quality codes (…F).
- AARP example: 35% of TRUE lines were Category II quality codes. Line coverage was 20.5%, while claim coverage was 55.3%.
- **Action:** fix the label before training (only PA-relevant lines TRUE).

### 7.2 `LINE_OF_BUSINESS` is unreliable
For payor = MEDICARE, 1.21M of 1.38M rows have no value, and the rest are mixed (Commercial, Blue Cross, Medicaid). It is likely the encounter's financial class. Don't use it as evidence of plan type.

### 7.3 The rules engine identifies payors by plan code
The Audit Service document (`US322761_AuditService_Auth_Messages`) shows the APE request carries a `planCode`, and each payor script reports either "plan code X matched one or more rules" or "no rules set up for plan code X". **The plan code → payor mapping already exists inside the rules scripts.** If obtained, it would replace or cross-check this name-based mapping.

### 7.4 CMS WISeR (January 2026)
A new CMS model adds PA to traditional Medicare in **NJ, OH, OK, TX, AZ, WA** for services on or after 15 January 2026 (e.g. knee arthroscopy, epidural steroid injections, nerve stimulators, skin substitutes). For 2026 NJ Medicare claims, a PA rule can depend on the **patient's state**.
- FAQ: https://www.cms.gov/priorities/innovation/files/document/wiser-model-frequently-asked-questions
- Operational guide: https://www.cms.gov/priorities/innovation/files/wiser-provider-supplier-guide.pdf

---

## 8. How this supports model training

Per the PRD, rules work well for coded lists but not for non-coded ones. The master sheet makes that split usable:

| Segment | Share of linked volume | Role |
|---|---|---|
| Coded-list payors | ~53% | Rules apply. The claims labels agree with them, which supports using claims labels as the model target |
| Non-coded payors | ~46% | **The model predicts** |
| Set aside | ~14% of all claims | Scope to agree |

**Uses of the master sheet**
1. **Routing:** decide whether a claim or request goes to the rules or to the model.
2. **Label checking:** compare rule vs label in the coded segment (confirmed positives, required but no evidence, evidence but not required).
3. **Features:** `rules_payor`, `rule_status` (on list / coded but not listed / no coded list / not applicable / unmatched), `match_level`, line of business, state.
4. **Evaluation:** a hide-the-rules test. Treat coded payors as non-coded, then measure how well the model reproduces the rules.
5. **Production:** translate new request payor names the same way; unseen names fall back to "unmatched".

---

## 9. Assumptions and limitations

| # | Item | Confidence | How to confirm |
|---|---|---|---|
| 1 | Anthem HealthPlus (NY Medicaid) → `EMHECH` (commercial code); no separate Anthem NY Medicaid code in the rules | Low | Rules team |
| 2 | Oscar → `NYOSCR` (assumed NY) | Medium | Patient ZIP check |
| 3 | `CMSHSP` codes beyond the 51 OPD codes come from other CMS programmes | Medium | Rules team |
| 4 | Mapping is at **payor** level, not plan level | Known limit | Plan-code lists |
| 5 | Rules file is a 2026 snapshot; earlier years may differ | Known limit | Use 2026 claims for validation |
| 6 | The 45.6% with no coded list cannot be validated with CPTs | Known limit | Coded lists or plan codes from the rules team |
| 7 | Long-tail 2.3% unmatched | Low impact | Map if required |

---

## 10. Open questions for other teams

**Rules / APE team**
1. Can we get the plan-code lists inside the payor scripts (`Authorization/Scripts/Integris/*.xml`)?
2. How do script names (e.g. "Aetna Medicare Adv.xml") map to the Excel `PAYOR_CODE`s?
3. What does `CMSHSP` contain beyond the CMS OPD list? Is WISeR applied by state?
4. Is the `ReferenceError: _response is not defined` seen for Anthem BCBS Colorado and Cigna Commercial a demo-only issue?

**Data team**
5. Which claims column holds the APE `planCode`? (Candidates: `BUCKET_PLAN_ID`, `PRODUCT_ID`.)
6. Does the Data Warehouse keep APE results (`authRequired` per procedure) per account for 2026?
7. Which 7 CARC/RARC codes are used in `GROUP_LABEL`? Are the pending codes (62, 288, 302, N640, 243, N473) signed off?

---

## 11. How to reproduce

In the SageMaker notebook, with `path`, `s3`, `rules_df` and `exploded_rules_v2` loaded:

```python
base = "https://raw.githubusercontent.com/Sidhartha10koneru/pa-scripts/main/"
for f in ["Payor_tag", "run_mapping.py", "validate_mapping.py"]:
    !wget -q -O {f} {base}{f}
%run -i run_mapping.py        # -> payor_master_review.xlsx
%run -i validate_mapping.py   # -> mapping_validation.xlsx
```

Always run `run_mapping.py` before `validate_mapping.py` in the same kernel. The validator reuses the `pairs` table already in memory.

---

## 12. Next steps

1. **Fix the label:** stop companion lines inheriting TRUE (`fix_labels.py`).
2. **Send the open questions** in section 10 to the rules and data teams.
3. **Confirm assumptions 1–3** in section 9.
4. **Build features** from the master sheet plus CPT category, payor × CPT history, diagnosis and place of service.
5. **Train and evaluate:** hide-the-rules test on coded payors, held-out non-coded claims, and a rules-only baseline.

---

### References
- CMS – Prior Authorization for Certain Hospital OPD Services: https://www.cms.gov/data-research/monitoring-programs/medicare-fee-service-compliance-programs/prior-authorization-pre-claim-review-initiatives/prior-authorization-certain-hospital-outpatient-department-opd-services
- CMS – OPD PA code list (PDF): https://www.cms.gov/files/document/opd-services-require-prior-authorization.pdf
- CMS – WISeR Model FAQ: https://www.cms.gov/priorities/innovation/files/document/wiser-model-frequently-asked-questions
- CMS – WISeR Provider and Supplier Operational Guide: https://www.cms.gov/priorities/innovation/files/wiser-provider-supplier-guide.pdf
- UHC Commercial PA requirements (3/2026): https://www.uhcprovider.com/content/dam/provider/docs/public/prior-auth/pa-requirements/commercial/UHC-Commercial-Advance-Notification-PA-Requirements-3-2026.pdf
- UHC Oxford PA requirements (7/2026): https://www.uhcprovider.com/content/dam/provider/docs/public/prior-auth/pa-requirements/oxford/Oxford-Prior-Auth-7-1-2026.pdf
- EmblemHealth preauthorization list: https://emblemhealth.com/providers/clinical-corner/um-and-medical-management/pre-authorization-list
- MVP UM policy guide (NY): https://content.mvphealthcare.com/provider/documents/mvp-utilization-management-policy-guide-new-york-state.pdf
- Cigna precertification requirements: https://www-cigna-com.extwideip.cigna.com/health-care-providers/coverage-and-claims/precertification
