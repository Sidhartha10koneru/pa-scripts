# PA Model – False-Negative Analysis, Bucket 4 (Mixed Cells)

*Status: 4 Nov 2026 · Model: LightGBM, 5 features, trained 2022–Apr 2026, no filing-order filter (`p_lgb_full_2022`) · Test: May–Jul 2026 · Unit: service (CSN + CPT) · Operating point: 80% recall (cut-off 0.421)*

Companion code: `bucket4_checks.py` (one runnable cell per section). Dependencies in the same folder: `pa_base_adapter.py`, `fn_root_cause.py`, `cell_label_check.py`.

---

## 1. Summary

- The model misses **~10.5k PA services (FN)** at 80% recall. They were split into 5 symptom buckets. This report covers **bucket 4, mixed cells: 2,603 FN (25%)**.
- A mixed cell is a 5-feature cell (CPT + modifier + POS + payor + plan) with ≥ 20 training lines and a **training PA rate of 5–50%**. Every line in a cell gets the **same score**. If that score is below the cut-off, **every PA in the cell is missed**, whether or not anything changed between training and test.
- So the question for bucket 4 is not "what changed?" but **"what makes the PA lines different from the non-PA lines in the same cell?"**
- 2,244 FN (86% of the bucket, 300 cells) were analysed. Result:

| # | Problem | FN | Share | Type |
|---|---|---|---|---|
| 1 | Auth number sits on the whole claim, so services that rarely need PA themselves inherit it | 1,195 | 53% | **Label limitation** |
| 2 | One-off code denial (CARC 197) on routine care | 509 | 23% | **Partly data issue** + **label question** |
| 3 | Repeat-patient code denial | 293 | 13% | Same as 2, + patient denial history |
| 4 | NYCE PPO physical therapy: PA after 25 visits | 134 | 6% | **Data issue, not available at prediction time** |
| 5 | Own auth on the service (real PA) | 113 | 5% | 42 data issue · 14 change · 13 insufficient · 44 no solution |
| – | Not analysed (small tail cells) | 359 | – | Likely data insufficient |

**One line:** most mixed-bucket FN come from **how PA is recorded** (claim-level auths, administrative denials), not from missing clinical signal. The real pre-visit signals found are **visit type, provider, department/site, PCP match, and auth/denial history**.

---

## 2. Background

### 2.1 FN buckets (all 10,457 FN)
| Bucket | Rule (first match wins) | FN | Share |
|---|---|---|---|
| 1 Likely label noise | Quality F code, or E/M / lab on a visit with a PA "anchor" service (code-range heuristic) | 3,431 | 33% |
| 2 Rare | < 20 training lines in the cell | 2,656 | 25% |
| 3 Historically non-PA | Training PA rate < 5% | 1,675 | 16% |
| **4 Mixed** | **Training PA rate 5–50%** | **2,603** | **25%** |
| 5 Model miss | Training PA rate ≥ 50% | 92 | <1% |

Buckets are **symptom groups from heuristics**. The causes in this report come from evidence. For example, some passengers ended up in bucket 4 because the bucket-1 code ranges didn't recognise cardiology codes (93xxx) as anchors.

### 2.2 Why train-vs-test change does not explain bucket 4
- In the unexplained mixed cells, the training PA rate is a **median 20%** and the test rate a **median 2.5%**. Test is lower, not higher.
- Only **39 FN (2%)** sit in cells where the test rate is ≥ 2× the training rate.
- The training/test gap is mainly the **AMBIGUOUS-drop inflation** in 2025 labels. Inflation raises scores, so it causes **FP**, not FN. Without inflation these cells would score even lower and still be FN.

### 2.3 Method
Two questions per FN:
- **A. What made it TRUE?** (post-visit evidence: auth number or denial code). This tells us whether the label is valid.
- **B. If the PA is real, was there a pre-visit signal?** This tells us data issue / insufficient / no solution.

Only information available **at prediction time** can become a feature. Post-visit information (remit codes, auth numbers, charges) can only be used to fix **labels** or to explain.

Script `fn_root_cause.py` ran these checks for each FN:

| Check | Data | Question |
|---|---|---|
| T1 label source | Test line | Auth number, PA denial code, or another remit row? |
| T2 shared | Other lines of the same visit (CSN) | Is the same auth number on another CPT? |
| T3 open auth | Patient history before the FN date | Was this auth number used at an earlier visit? |
| T4 subgroup | Cell training lines (2025 → Apr 2026, primary, labelled) | Does a pre-visit value (specialty, department, provider, PCP, visit type, primary dx, visit number) have a training PA rate ≥ cut-off? |
| T5 change | Cell training vs test rate | Did PA rise ≥ 2×? |
| T6 too few | Cell training PA count | Fewer than 10 PA? |

The problems below come from splitting the FN on **T1 (label source)** and **T2 (shared or not)**:

```
2,244 FN
├── TRUE from a denial code / other remit row (no auth) ── 802
│     ├─ same patient code-denied before (≥45 days) ──── Problem 3 (293)
│     └─ not denied before ──────────────────────────── Problem 2 (509)
└── TRUE from an auth number ─────────────────────────── 1,442
      ├─ same auth on another service of the visit
      │     ├─ PT/OT codes (NYCE PPO) ───────────────── Problem 4 (134)
      │     └─ drug / imaging / all lines of the visit ─ Problem 1 (1,195)
      └─ only this service has it ───────────────────── Problem 5 (113)
```

---

## 3. Label-rule review (labelling script)

The production labelling script was checked rule by rule on a 2026 sample (first ~2.05M lines, 30,556 Level-1 TRUE). *Code: `bucket4_checks.py` §3.*

| Auth value type | FALSE | TRUE | UNCONFIRMED | Read |
|---|---|---|---|---|
| Number/code | 81 | **30,230** | 338 | 98.9% of TRUE come from auth-number-like values |
| No-auth / placeholder text (NO AUTH REQUIRED, NPN, NPAR…) | 521 | **0** | 15 | ✅ Never TRUE |
| Referral text (PAPER REFERRAL…) | 38 | **0** | 104 | ✅ Never TRUE; falls back to codes |
| Other text | 1,442 | 9 | 86 | ✅ The 9 are APPROVED-type confirmations, by design |
| Dates (MMDDYYYY) | 0 | **317** | 0 | ⚠️ ~1% of TRUE; 28 equal the service date |

Minor cleaning items (≈1% of TRUE, **not FN drivers**):
- Level 2 `contains("M62|N54")` also matches **N549 / N548**: 66 lines in the sample.
- Date values typed into the auth field: 317 TRUE, 28 identical to the service date.
- One auth value shared by more than 5 patients: 1 value.

**Rules work as designed.** The real label issues are structural (Problem 1, and the AMBIGUOUS drop).

---

## 4. Problems

### Problem 1: the auth is on the claim, not the service (1,195 FN, 53%)

**What it is.** When a visit has an auth, the **same auth number is written on every service of that claim**. The labelling rule makes every one of them TRUE, including services that don't need PA themselves.

**Evidence (2026 sample).** *Code §4.1.*
- Visits with ≥ 2 CPTs and an auth: 7,349.
- **91.7%** have an auth on every line; 8.3% have some lines blank.
- Only **0.2%** have 2+ different auth values on one visit.

**Which codes inherit it.** Rate of TRUE when a code is alone on the claim vs alongside other services. *Code §4.2.*

| Code | Alone | With others | × | Read |
|---|---|---|---|---|
| G2211 (E/M add-on) | 0.1% | 0.8% | 9.8 | Almost never TRUE on its own |
| 93000 (ECG) | 0.1% | 0.9% | 6.3 | Almost never on its own |
| 36415 (blood draw) | 0.3% | 1.4% | 3.9 | Almost never on its own |
| 99213 / 99214 / 99215 | 0.5–1.2% | 1.2–3.5% | 2.3–2.9 | Rarely on its own |
| 96372 (injection admin) | 4.5% | 10.3% | 2.3 | Mostly carried by the drug |
| 93306 (echo) | 10.4% | 13.0% | 1.25 | **Needs PA itself** |
| 93015 (stress test) | 12.4% | 31.1% | 2.5 | **Needs PA itself** |
| 97112 (PT) | 0.9% | 12.3% | 13.6 | Not comparable (PT visits always bill several therapy codes) |

**Example.** Injection 96372 + pegfilgrastim Q5111 on the same visit, both carrying the same auth (52 FN). The drug needs PA. The injection code inherits it. Most 96372 lines (flu shot, B12…) have no auth, so the cell scores low and these are FN.

**Why the model misses it.** The 5 features don't include "what else is on the visit".

**Type: label limitation.** From claims we can't tell which service on a claim the auth was for.

**Possible experiments (later):**
- Data-derived list of "rarely own-PA" codes (alone-rate below a threshold): treat their claim-carried TRUE as not-PA, then measure FN/FP.
- Visit-level scoring (the visit-level label is reliable). The team asked for service level, so this is an option to raise.
- "Other services on the visit" as a feature, **only if** available at prediction time (AccuReg ACCUREGCPT; to confirm with the team).

---

### Problem 2: one-off code denials (509 FN, 23%)

**What it is.** No auth number. TRUE only because the remit carried a PA denial code, mostly **CARC 197** ("precertification/**notification**/authorization absent"). The patient had **no earlier code-denial**.

**Evidence.** *Code §4.3, §4.4.*
- 360 TRUE from a denial code (197 dominant); 149 TRUE from **another remit row** while the first row shows normal payment (45 / 3 / 1). These 149 are the weakest labels.
- Services are routine: office visits, preventive exams (99395/99396), blood draws, urinalysis, eye exams. Cells are a median 2.5% PA in test.

**TRUE vs FALSE lines in the same cell** (code-denied vs paid), top cells:

| Cell | Denied vs paid | Provider is patient's PCP (denied vs paid) | What stands out in denied lines |
|---|---|---|---|
| 36415 Emblem HIP | 86 vs 7,840 | **14% vs 55%** | Specialists ordering (Neurology 11.7×, GI 5.8×); new patients 5.9× |
| 99213 Emblem HIP | 123 vs 9,013 | **12% vs 29%** | Dermatology / ENT / podiatry departments (4–41×) |
| 99214 Emblem HIP | 179 vs 10,411 | **10% vs 30%** | Ophthalmology 17×, GI 8.5×, Ob-Gyn 6.5× departments |
| 36415 MVP | 231 vs 4,432 | 36% vs 45% | One lab-draw department (Cranberry Dr) **34×** |
| 99213 MVP | 393 vs 4,782 | 16% vs 19% | One urgent-care department (360 N Bedford Rd) **26×** |
| 99396 Anthem Commercial | 104 vs 728 | 75% vs 40% (opposite) | Internal-medicine "Physical" visits at a few primary-care departments (317 E 34th St **26×**); Ob-Gyn wellness almost never denied |

**Read.**
- **Emblem HIP (HMO):** denials happen when someone **other than the PCP** bills. That's consistent with a missing referral.
- **MVP:** specific departments/sites have very high denial rates. Small counts (10–30 lines); possibly a network/credentialing issue (inference).
- **Anthem 99396:** internal-medicine physicals at specific primary-care sites. Not referral-related.
- The PCP mismatch raises the denial rate ~4–7×, but only to **~2%**, still far below the cut-off. Site-level rates reach 60–80% but on small counts.

**Type.**
- **Partly data issue:** department/site and provider-vs-PCP are pre-visit (HL7 PV1 location; ROL PP = PCP, AT = attending).
- **Label question:** a one-off 197 on routine care looks administrative (referral/site), not clinical PA.
- Remaining one-offs with no signal: **no solution with current data**.

---

### Problem 3: repeat-patient code denials (293 FN, 13%)

**What it is.** Same as Problem 2, but the **same patient had an earlier code-denial ≥ 45 days before** (144 on the same CPT, 149 on another).

**How "earlier denial" is derived** (analysis variable, not a model feature). *Code §4.3.* For an FN (patient P, date D): take all of P's lines in the 2025–2026 std files (any CPT, any visit). A line is a code-denial if it is TRUE with no auth number. If one is dated ≤ D − 45 days, prior_denial = yes.

**Evidence.**
- 802 code-labelled FN, 547 patients. No earlier denial: 509 (63%); earlier, different CPT: 149 (19%); earlier, same CPT: 144 (18%).
- Repeat share by plan: Emblem HIP 74/117 (63%), Emblem Medicaid 26/42 (62%), BCBS Arizona 10/18 (56%), Anthem Commercial 73/181 (40%), MVP 38/175 (22%).
- Anthem 99396 cell: patients ever denied were denied almost every time (69 TRUE / 3 FALSE); never-denied patients 525 FALSE / 0 TRUE. The same patients' annual exams were denied year after year.

**Type.**
- **Data issue (provisional):** patient denial history predicts it, **if** claims history is available at prediction time (it is not in the HL7 feed).
- **Plus the same label question as Problem 2.**
- **Not yet known:** precision. How often is a patient with an earlier denial *not* denied next time?

---

### Problem 4: NYCE PPO physical therapy, PA after 25 visits (134 FN, 6%)

**What it is.** NYCE PPO is a new plan from 1 Jan 2026 (replacing GHI CBP). Policy **UMC0394**: PA required for "physical therapy and occupational therapy after 25 combined visits".

**Evidence.** *Code §4.5.*
- Cell 97112 + GP + POS 11 + NYCE PPO: 0 PA Jan–Mar 2026; April 19/50; test ~19%.
- First PA at **visit 26 for 26 of 32 patients**. After a patient's PA starts, every later visit is PA (138/138).
- All TRUE come from real auth numbers (clean labels).

**Why the model misses it.** Visit 5 and visit 30 have identical features.

**Type: data issue, not available at prediction time.** The driver is the patient's PT/OT visit count across **all providers**. At best we would know our own system's visits. Record it; don't add it as a feature. Test an own-system visit-count feature only if the team confirms a source.

---

### Problem 5: own auth on the service (113 FN, 5%)

**What it is.** The FN line has a real auth number and **no other service on the visit has it** (usually the only service on the claim). These are the most trustworthy PA labels in bucket 4. *Code §4.6.*

**Why the model still misses it.** The auth number is the **label**, not a feature: at prediction time there is no auth yet. A clean label doesn't make a service predictable. E.g. echo 93306 at Empire Plan is ~6% PA in test, and nothing in the 5 features separates the 6% from the 94%.

| Cause | FN | Example | Type |
|---|---|---|---|
| Unexplained | 44 | Echo 93306 (Empire Plan, Magnacare), 99213 Emblem HIP, upper endoscopy 43239 | No solution with current data |
| Auth open from an earlier visit | 24 | 96372 NYCE PPO: auth first used 125 days earlier (injection course) | Data issue, if auth history is available at prediction time |
| Subgroup rule | 18 | 93015 NYCE PPO: visit type "Stress Test Myocard Perf" = **92% PA**. 95806 Anthem BlueCard: one provider = **80% PA**. | Data issue: visit type and provider are pre-visit |
| Rate rose after training | 14 | – | Change |
| Too few PA in training | 13 | – | Data insufficient |

Note: the largest problem-5 cells are **Tricare West new-patient visits** (99203: 14, 99204: 9). Tricare Prime requires referral authorisations for specialty care, so these are likely referral auths. **Not verified.**

*Why some rows look inconsistent:* the check's "fit rate" uses 2025+, primary-only lines. The model trained on 2022+, all filing orders, with ≥ 100 lines per leaf. A cell can show a 53% fit rate here and still be an FN.

---

## 5. Caveats
- Label-rule and claim-level checks use the **first ~2M lines of 2026**, not a random sample. Rerun on full years before final numbers.
- T2 uses the std file's auth field (first remit row only).
- Patient history starts Jan 2025; older denials/auths are not seen.
- `PAID_AMT` is ~0 for nearly all lines, so "denied then paid later" could not be verified.
- 359 bucket-4 FN in small tail cells were not analysed.
- FN counts are measured on labels that carry the same biases; they will move after any label change.

## 6. Open questions for the team
1. **Target definition:** service level ("this service needs PA") or visit level ("this visit needs PA")? Do drug-administration codes, referral auths, and 197 denials on routine care count as PA?
2. **Available at prediction time?** Patient claims/denial history; prior auth numbers; own-system PT/OT visit counts; the AccuReg ACCUREGCPT list of services per visit; department / visit type / attending vs PCP from HL7.
3. **Ground truth source:** can we get auth records (authorised CPTs per auth number)? That would settle Problem 1 directly.

## 7. Candidate experiments (after the definitions are agreed)
| Problem | Experiment | Measure |
|---|---|---|
| 1 | "Rarely own-PA" code list from alone-on-claim rates → relabel claim-carried TRUE; or visit-level scoring | FN/FP at 80/90% recall vs benchmark |
| 2, 3 | Add department, visit type, provider-is-PCP (pre-visit) | Same |
| 3 | First: precision of "earlier denial". Then a history feature if available | Same |
| 4 | Own-system PT/OT visit count, only if a source exists | FN in NYCE PT cell |
| 5 | Visit type / provider features (shared with 2, 3) | Same |
| Labels | Exact-code match for N54/M62; auth value = service date → not an auth | Label counts |
