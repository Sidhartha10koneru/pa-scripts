# Payor Master Sheet – Review (v2)

*Prior authorization (PA) prediction project · updated 15 Oct 2026*

This replaces `PAYOR_MASTER_SHEET_REVIEW.md` (v1). New in v2: plan-level mapping, coded-list-only lists,
primary-payer filter, the Medicare evidence with sources, corrections to v1 figures, and definitions.

---

## 1. Why we need a master sheet

| Source | What it gives us | Problem |
|---|---|---|
| **Rules file** `REQ-260378.xlsx`, sheet *Plans with Services* | Payor PA policies: `PAYOR_CODE`, `PLAN_NAME`, CPT lists (`PLAN_CPT_CODES`), `CODING_STATUS` | Payor and plan names don't match the claims |
| **2026 claims** (parquet, 8.62M claim lines) | `PAYOR`, `PLAN`, `FINAL_CPT`, `GROUP_LABEL` (TRUE / FALSE / AMBIGUOUS), `FILING_ORDER_C`, `CSN` | No rules payor code |

There is **no shared ID** between the two. The master sheet is the lookup that connects them:

> claims `PAYOR` + `PLAN` → rules `PAYOR_CODE` (+ rules `PLAN_NAME` where plans differ)

Per the PRD:
- rules decide PA for payors with a **CODED_LIST**;
- the model predicts PA for everything else.

---

## 2. Definitions

| Term | Meaning |
|---|---|
| **Claim** | One bill (`CSN`). It usually has several lines. |
| **Claim line** | One billed code (CPT/HCPCS) on a claim. One row in the parquet. **All volumes in this doc are lines, not claims.** |
| **PA line** | A line with `GROUP_LABEL = TRUE`. TRUE comes from an auth-number pattern or 7 CARC/RARC codes. It is **labelled** PA, not confirmed PA. |
| **Companion line** | A line that rides along on a PA claim but never needs PA itself: office visits, Category II codes, blood draw, infusion administration, etc. The label is set per claim and copied to every line, so these lines inherit TRUE. They are excluded from scoring by the regex `^(99[2-4]\d\d\|\d{4}F\|36415\|96[3-5]\d\d\|Q996[5-7]\|A957[5-9]\|G2211)$`. |
| **Coverage** | Share of PA lines (companions excluded) whose code is on the payor's rules list. *How much PA does the list explain?* |
| **Lift** | (Share of PA lines on the list) ÷ (share of non-PA lines on the list). *Does the list separate PA from non-PA?* 1 = no signal; 5 or more = strong. |
| **Primary payer** | `FILING_ORDER_C = 1` (inferred, see §7). |

Coverage and lift are always read together. A list containing every code has 100% coverage but a lift of 1.

---

## 3. How the master sheet was built

1. **Distinct pairs.** We took every distinct claims `PAYOR` + `PLAN`: 2,437 pairs.
2. **Name patterns.** An ordered list of regex patterns maps names to a rules `PAYOR_CODE`.
   - The first match wins, and specific brands come before their parent company (e.g. MERITAIN before AETNA).
   - Patterns were written and reviewed by hand: the script suggests candidates, and a person decides.
   - Every code is checked to exist in the rules file.
3. **Overrides.** These are applied last, on "PAYOR | PLAN", for cases where the plan name changes the answer:
   - Medicare supplement / Medigap → `SUPPLEMENT_NO_PA`
   - BlueCard → `BLUECARD`
   - Federal Employee Program → `FEP`
   - Optum Care Network → `DELEGATED`
   - UHC Medicaid → `UHCNY`
   - Presbyterian → `PRESBY`
   - NM BCBS plan names → `NMBCBS`
   - Anthem Medicare → `MEDADV`
4. **State from ZIP.** When a name has no state, patient ZIP3 decides it (e.g. "ANTHEM" claims are New York → `EMHECH`).
5. **Plan level.** For 4 payors whose rules plans have different lists, a `rules_plan` column is added (§5).
6. **Validation.** Every mapping is scored against the claims labels (§8).

So every mapping is a **name-based judgement tested with data**, not an exact key match.

### Categories (all 8.62M claim lines)

| Category | Lines | Share |
|---|---|---|
| Linked to a rules payor | 7,197,657 | 83.5% |
| NOT_IN_RULES (payor not in the rules file) | 416,830 | 4.8% |
| SUPPLEMENT_NO_PA (pays after Medicare) | 386,222 | 4.5% |
| BLUECARD (out-of-area Blue member, home plan rules apply) | 319,578 | 3.7% |
| UNMATCHED (not mapped yet) | 196,584 | 2.3% |
| FEP (Federal Employee Program) | 98,670 | 1.1% |
| DELEGATED (medical group runs its own auth) | 6,028 | 0.1% |

---

## 4. Coded lists only

Only rules rows with `CODING_STATUS = CODED_LIST` give a payor a PA list. Lists are built from `PLAN_CPT_CODES`, keeping 5-character tokens.

Rules payors with no coded plan are still mapped, because the code is useful as a model feature, but they get `rules_plan = NO_CODED_PLAN`.

Examples: Emblem GHI and HIP, NYSHIP (Empire Plan), Anthem BCBS NY, MVP, 1199, Magnacare, Moda, Horizon commercial, NM Medicaid.

---

## 5. Plan-level mapping

**Question from the team:** one payor has many plans in the claims. Do they all have the same PA rules?

**Check:** for each rules payor with more than one coded plan, we compared the CPT lists (share of codes common to all plans).

| Rules payor | Claim lines | Coded plans (codes) | Share common | Plan level needed? |
|---|---|---|---|---|
| UHC | 826,872 | Commercial (2,867) / Medicare Advantage (1,066) | **0.29** | **Yes** |
| AETNA | 620,828 | 3 regional plans (1,395 each) | 1.00 | No |
| NJBCBS (Horizon) | 95,549 | 2 Medicare plans | 1.00 | No, but see below |
| ORRBCS (Regence OR) | 91,415 | Commercial / Uniform / FEP ×2 / Medicare | **0.06** | **Yes** |
| PSHLTH (PacificSource) | 74,254 | 3 plans | 0.87 | Mostly no |
| NMBCBS | 73,317 | Commercial (1,063) / Medicare Advantage (895) | **0.14** | **Yes** |
| AZBCBS | 36,600 | 10 employer plans | 0.03 | Yes, but no plan info in claims; left at payor level |

### Plan rules applied (`plan_mapping.py`)

| Payor | Claims "PAYOR \| PLAN" contains | → Rules plan |
|---|---|---|
| UHC | MEDICARE, ADVANTAGE, AARP, DUAL, SNP, MA | UnitedHealthcare Medicare Advantage |
| UHC | anything else | UnitedHealthcare Commercial |
| ORRBCS | MEDICARE, ADVANTAGE | Oregon Regence BCBS Medicare |
| ORRBCS | UNIFORM, UMP | Oregon Regence BCBS Uniform Medical |
| ORRBCS | anything else | Oregon Regence BCBS Commercial Individual and Fully Funded |
| NMBCBS | TURQUOISE, CENTENNIAL, MEDICAID | NO_CODED_PLAN (NM Medicaid) |
| NMBCBS | MEDICARE, MED ADV, ADVANTAGE, DUAL | NM BCBS Medicare Advantage |
| NMBCBS | anything else | NM BCBS Commercial |
| NJBCBS | MEDICARE, TOTALCARE, ADVANTAGE, DUAL | ALL (both Medicare plans share one list) |
| NJBCBS | anything else | NO_CODED_PLAN (Horizon commercial has no coded list) |
| all others | – | ALL (the payor's plans share one list) |

### Example: why UHC must be split
A code that is only on UHC Commercial's list:

| Claim | Payor level only | By plan |
|---|---|---|
| UHC employer plan | needs PA ✅ | needs PA ✅ |
| UHC Medicare Advantage (AARP) | needs PA ❌ wrong | no PA ✅ |

Without the split, Medicare Advantage claims are checked against about 2,000 commercial-only codes.

| UHC claims (primary payer) | Lift, payor level only | Lift, by plan |
|---|---|---|
| UHC Commercial | 5.2 | **11.6** |
| UHC Medicare Advantage | 4.7 | **11.9** |

Medicare (CMSHSP) has **one** coded plan, so there is nothing to split (see §9).

---

## 6. Rules text: conditions and plan hints (checked, minor)

- **Conditions in ` Auth Requirement(Direct to Customer)`.**
  - 8.4% of coded rows carry a condition (mostly the UHC family).
  - Diagnosis conditions are in 211 rows (1.7%) and touch 4% of listed lines.
  - CPT + diagnosis gave the **same** lift as CPT alone, so diagnosis is left to the model.
- **Plan names in the rules text.**
  - Only SNP / D-SNP / Dual Complete match claims plans: about 46k lines (UHC ~25k, Aetna ~18k).
  - Group numbers don't overlap with the claims.
- **`PAYOR_ALIAS`** is one value per payor, so it isn't a plan ID.

---

## 7. Primary payer filter

| `FILING_ORDER_C` | Lines | Share | TRUE rate |
|---|---|---|---|
| 1 | 7,211,372 | 83.7% | **0.4%** |
| 2 | 1,380,419 | 16.0% | 0.1% |
| 3 | 23,383 | 0.3% | 0.0% |
| 4 | 210 | <0.1% | 0.0% |
| 99 | 6,185 | 0.1% | 0.1% |

- Filing order 1 carries 4× the PA rate, consistent with **1 = primary payer**. This is inferred; the data team still needs to confirm it.
- PA is obtained from the primary payer. On secondary claims, that payer's list doesn't apply.
- **Decision:** rules and validation use primary-payer lines only (`PRIMARY_ONLY = True`).
- Medicare as payor: 1,388,595 lines at filing order 1 and 9,098 at filing order 2.

---

## 8. Validation

**Method (`validate_mapping.py` v3):**
- TRUE/FALSE lines only, primary payer only.
- Each claims payor is scored against its **own** mapped list, coded rows only and at plan level.
- Status rules:
  - **FITS:** lift ≥ 5 and coverage ≥ 25%
  - **WEAK_FIT:** lift ≥ 2
  - **DOES_NOT_FIT:** lift < 2
  - **NO_CODED_LIST:** the mapped list is empty
  - **NOT_TESTABLE:** fewer than 20 PA lines

### Status (share of scored lines)

| Status | All filing orders (5.74M lines) | **Primary only (5.13M lines)** |
|---|---|---|
| FITS | 45.4% | **29.5%** |
| WEAK_FIT | 6.8% | **25.4%** |
| DOES_NOT_FIT | 0.1% | 0.1% |
| NO_CODED_LIST (model segment) | 47.4% | **44.6%** |
| NOT_TESTABLE | 0.3% | 0.4% |

The drop in FITS is almost entirely **Medicare**, about 20% of scored lines, moving from FITS to WEAK (see §9). The other large payors were unchanged or slightly better.

### Key payors (primary only)

| Claims payor → rules | Lines | PA lines | List size | Coverage | Lift | Status |
|---|---|---|---|---|---|---|
| Medicare → CMSHSP | 1,029,988 | 87 | 581 | 8.3% | 22.7 | WEAK |
| Aetna → AETNA | 331,916 | 536 | 1,395 | 69.1% | 27.9 | FITS |
| United Healthcare → UHC Commercial | 327,305 | 1,914 | 2,867 | 45.4% | 11.6 | FITS |
| UHC Medicare Advantage → UHC MA | 251,363 | 1,919 | 1,066 | 58.2% | 11.9 | FITS |
| Cigna → CIGNA | 220,237 | 887 | 1,622 | 59.6% | 42.7 | FITS |
| BCBS of Oregon → Regence Commercial | 30,995 | 1,026 | 1,639 | 36.8% | 30.1 | FITS |
| Oxford → OXFORD | – | – | 3,161 | 36.2% | 3.8 | WEAK (broad list) |

"PA lines" includes companion lines; coverage and lift exclude them.

### Largest payors by volume (primary, TRUE+FALSE lines)
Medicare 1.03M (~20%), Emblem GHI 659k, Aetna 332k, UHC Commercial 327k, UHC Medicare Advantage 251k, Cigna 220k, Empire Plan 220k, Anthem BCBS NY 207k.

---

## 9. Medicare: questions from the team

### Q1. How do we know claims "MEDICARE" is the plan the rules file means?
**There's no shared ID. We inferred it and tested it.**

| Evidence | Shows |
|---|---|
| Claims payor is just "MEDICARE" (no company name) | Original Medicare. Medicare Advantage carries a company name (UHC, Humana, Aetna, …) and is mapped to that company. |
| The rules file has one Original Medicare payor: `CMSHSP`, single plan "CMS Medicare" | No other candidate |
| CMSHSP's coded list contains **all 51** codes on CMS's hospital outpatient PA list | Only a Medicare list contains that exact set |
| On primary Medicare lines, listed codes are **22.7×** more common on PA lines | The list behaves like Medicare's list in our data |

**Still to confirm:** the rules team should tell us which `planCode` the rules engine uses for Original Medicare.

### Q2. Are those codes definitely Medicare?
- The **list** is Medicare's. It is the CMS programme "Prior Authorization for Certain Hospital Outpatient Department Services", applied nationally since 1 July 2020, created in the CY2020 OPPS final rule (CMS-1717-FC), for Medicare fee-for-service.
- The **individual codes** are not exclusive to Medicare: blepharoplasty, Botox, vein ablation, etc. also need PA from commercial payors.
- So the evidence is the **full set** of 51 codes being on CMSHSP, not any single code.

### Q3. Different Medicare plan names and line-of-business values. Why one code?
1. **The PA programmes are national.**
   - The outpatient PA applies nationally, and regional Medicare Administrative Contractors (MACs) only process claims for their geographic jurisdiction [CMS – What's a MAC].
   - Railroad Medicare patients follow the same outpatient PA, with requests going to the jurisdictional MAC [Palmetto GBA].
   - **Exception:** WISeR (NJ, OH, OK, TX, AZ, WA from Jan 2026, Original Medicare only).
2. **Claims plan names** ("Part A&B", "Part B", "NGS NY", "Railroad") read as billing-contractor or Medicare-part variants.
   - "NGS" matches National Government Services, the NY contractor.
   - **"Part B OR" is unexplained.** This is our interpretation of the names, not a sourced fact.
3. **The rules file has only one Medicare plan**, so there is nothing to split by.
4. **`LINE_OF_BUSINESS` is not used.**
   - It behaves like the account financial class, not the insurance plan.
   - Medicare rows marked Commercial or Medicaid were mostly secondary-payer cases, now removed by the primary filter.
5. **We split only where the rules show different lists** (UHC, Regence, NM BCBS).
6. **Untested:** Part B, NGS, Part B OR and Railroad each have 0–2 PA lines.

**Where "same rules" is too strong:**
- the outpatient PA applies only in hospital outpatient departments (type of bill 13X), not in doctors' offices or ASCs;
- WISeR depends on the state;
- local coverage policies differ by MAC (these govern coverage, not PA).

### Correction to v1
- v1 reported Medicare coverage ~22.7% and lift ~140. That **included claims where Medicare was secondary**: 43 of 130 PA lines, mostly on the list.
- On those claims, the PA came from the primary payer and the auth number was copied onto the Medicare crossover claim.
- **Use the primary-only figures:** 87 PA lines in about 1M, coverage 8.3%, lift 22.7.

### What the Medicare numbers mean
- **Label:** 87 TRUE vs 1,029,901 FALSE lines (primary, AMBIGUOUS excluded).
- Original Medicare rarely requires PA, which fits a very low TRUE rate. Most of the 87 are likely label noise. **Not yet verified** (check in §12).
- Possible gap: about 0.4% of FALSE Medicare lines have a listed code. These may be missed auths, or services where PA didn't apply (setting or state).
- **We do not relabel from the rules.** We keep a separate `rules_says_pa` feature and review the disagreements.

Sources:
- [CMS – What's a MAC](https://www.cms.gov/medicare/coding-billing/medicare-administrative-contractors-macs/whats-mac)
- [CMS operational guidance – PA for certain hospital OPD services](https://www.plasticsurgery.org/documents/Health-Policy/HOPD-Prior-Auth-FAQ.pdf)
- [CGS – OPD Prior Authorization](https://www.cgsmedicare.com/parta/pa/opd.html)
- [Palmetto GBA Railroad Medicare – OPD PA](https://dmepdac.com/palmetto/rr.nsf/DID/B8KUUO5DMO)
- [ABA – The WISeR Model Comes to Original Medicare](https://www.americanbar.org/groups/health_law/resources/esource/2026/the-wiser-model-comes-to-original-medicare/)

---

## 10. Other payor decisions

| Payor | Decision | Why |
|---|---|---|
| AARP | → UHC (plan level: Medicare Advantage) | AARP plans are insured by UHC. The UHC list fits AARP PA lines best, and UHC Medicare Advantage and AARP lists have the same size (1,066). |
| AARP MEDICARE | Keep on UHC Medicare Advantage; DOES_NOT_FIT, low volume | All its plans are Medicare Advantage. Only ~350 scored lines. The misfit comes from one plan (Complete PPO, 197 lines). |
| UHC Medicaid names ("UHC Medicaid Cap Alt Payer", "UNITED HEALTHCARE MEDICA") | → UHCNY | Were wrongly on the UHC Commercial list (fixed) |
| Presbyterian Turquoise | → PRESBY | Was caught by the TURQUOISE → NMBCBS pattern (fixed) |
| BCBS OF NM | NM Commercial, WEAK (lift 2.0) | Only one plan name ("BCBS NM"), so it can't be split; 0.2% of lines |
| Horizon commercial | NO_CODED_PLAN | Horizon's coded plans are Medicare only |
| Oxford | WEAK (3.8) | Unchanged with coded-only lists; the list is broad |
| Optum Care Network | DELEGATED | The medical group runs its own auth |

---

## 11. Assumptions and limitations

1. Mapping is by name, plus ZIP for state. There is no shared ID with the rules engine.
2. `FILING_ORDER_C = 1` is assumed to mean primary payer (inferred from the PA rate).
3. Label TRUE = auth number or CARC/RARC present. It is claim-level and copied to every line; companion lines are excluded by a fixed code list.
4. `LINE_OF_BUSINESS` is not used (unreliable).
5. AZ BCBS (10 employer plans) and BCBS NM can't be split by plan from the claims.
6. Volumes are claim lines, not distinct claims.
7. Unmatched 2.3% is left to the model.

## 12. Open questions

**Rules team**
- Which `planCode` is used for Original Medicare, UHC Commercial vs Medicare Advantage, and Regence / NM BCBS plans?
- Does CMSHSP include WISeR and DMEPOS codes, and are they state-specific?

**Data team**
- Confirm `FILING_ORDER_C` (1 = primary?).
- Meaning of `LINE_OF_BUSINESS`.
- Where auth numbers are captured for Medicare.

**Our checks (pending)**
- Medicare's 87 TRUE lines: companion share and top codes.
- "Rules say PA, label says no" counts per payor.
- Distinct-claim (`CSN`) counts per payor.

## 13. Next steps
1. Freeze the master sheet (primary only, coded lists, plan level).
2. Fix the labels: keep TRUE only on the line that needed PA, not on companion lines.
3. Build features: `rules_payor`, `rules_plan`, `rules_says_pa`, CPT group, diagnosis, filing order, state.
4. Train on the non-coded segment, with a rules-only baseline and a held-out non-coded test.

---

## 14. Reproduce

Scripts in `Sidhartha10koneru/pa-scripts`:

| Script (repo name) | Notebook name | Does |
|---|---|---|
| `Payor_tag` | – | Helper functions (`apply_keywords`, …) |
| `run_mapping.py` | `run_mapping.py` | Name patterns + overrides → `pairs`, `payor_master_review.xlsx` |
| `plan_mapping.py` | `plan_map.py` | Adds `rules_plan` |
| `validate_mapping.py` (v3) | `validate_map1.py` | Coverage and lift per payor and plan → `mapping_validation.xlsx` |

```python
# needs: s3, path, df_cols (claims), rules_df (sheet "Plans with Services")
%run -i run_mapping.py
%run -i plan_map.py          # must run after run_mapping (it rebuilds `pairs`)
# set PRIMARY_ONLY = True at the top of validate_map1.py
%run -i validate_map1.py
```
