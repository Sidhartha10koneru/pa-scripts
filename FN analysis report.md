# PA Model – Why the Model Misses PA (False-Negative Analysis) and Top 3 Experiments

*Status: 4 Nov 2026 · Model: LightGBM, 5 features (CPT, modifier, place of service, payor, plan), trained 2022–Apr 2026, no filing-order filter · Test: May–Jul 2026 · Unit: service (CSN + CPT) · Operating point: 80% recall (cut-off 0.421)*

Detailed bucket-4 write-up: `FN_bucket4_mixed_cells_report.md` · Reproducible checks: `bucket4_checks.py`, `fn_root_cause.py`

---

## 1. Summary

- At 80% recall the model misses **10,457 PA services (FN)**. We analysed **7,742 (74%)** and traced each one to where its PA label came from, and whether anything known before the visit could have predicted it.
- **Main finding: most FN are not missing clinical signal. They come from how PA is recorded on claims.**
  - **The auth number sits on the whole claim, not the service.** Every service on an authorised claim becomes PA, including quality-reporting codes, office visits and labs that never need PA on their own. **67% of analysed FN**; a quarter of all analysed FN are quality codes alone.
  - **17%** are TRUE only from a payer denial code (mostly 197 "auth absent") on routine care, which looks administrative (referral/site) rather than clinical PA.
  - **11%** are therapy visit-count rules (NYCE PPO confirmed; Cigna to verify). The visit count isn't available at prediction time.
  - Only **~5%** are clean PA the model could, in principle, learn better: rare/new cells and a model setting.
- **Top 3 experiments**, cheapest and clearest first:
  1. Remove quality codes from scope.
  2. Let the model fall back on what it knows about the CPT for rare/new cells.
  3. Stop counting claim-carried auths on codes that don't need PA themselves.

---

## 2. How the FN were analysed

**Step 1 – symptom buckets.** Each FN was grouped by what its cell (the 5 features) looked like in training:

| Bucket | Rule | FN | Analysed |
|---|---|---|---|
| 1 Likely label noise | Quality code, or office visit / lab on a visit with a PA service | 3,431 | 2,744 |
| 2 Rare | < 20 training lines in the cell | 2,650 | 1,332 |
| 3 Historically non-PA | Training PA rate < 5% | 1,675 | 1,330 |
| 4 Mixed | Training PA rate 5–50% | 2,603 | 2,244 |
| 5 Model miss | Training PA rate ≥ 50%, still scored low | 92 | 92 |

**Step 2 – root cause per FN.** Two questions:
- **A. What made it PA?** An auth number, or a payer denial code? Is the same auth on another service of the same visit?
- **B. If the PA is real, was there a signal before the visit?** Visit type, provider, department, history.

Only information available **at prediction time** can become a model feature. Post-visit information (auth numbers, remit codes) can only fix **labels** or explain.

**Key facts behind the analysis:**
- On visits with 2+ services and an auth, **91.7%** carry the same auth on every line; only 0.2% have different auths per service.
- The production labelling rules work as designed: no-auth text and referral text never become PA. Minor leaks are ~1% of TRUE (date values in the auth field; N549 matching the N54 rule).

---

## 3. What we found

### 3.1 Combined (7,742 FN)
| Problem | B1 | B2 | B3 | B4 | **Total** | Share | Type |
|---|---|---|---|---|---|---|---|
| **Claim-level auth** (auth on the whole claim) | 2,705 | 701 | 608 | 1,195 | **5,209** | **67%** | Label / scope |
| ↳ of which quality codes (CPT II, `####F`) | 1,930 | – | – | – | 1,930 | 25% | Scope: never PA |
| **Code denial** (TRUE from denial code 197, no auth) | 32 | 204 | 258 | 802 | **1,296** | 17% | Label question + partly data |
| **Therapy visit rules** | – | 296 | 408 | 134 | **838** | 11% | Data not available at prediction time |
| **Own auth** (real PA, nothing shared) | 7 | 131 | 56 | 113 | **307** | 4% | Rare cells / no signal |
| **Model setting** (bucket 5) | | | | | **92** | 1% | Model |

### 3.2 Per bucket, in one line each
- **Bucket 1 (label noise):** 98.6% claim-level auth. 1,930 are quality codes (blood-pressure and medication-list reporting codes) riding on authorised claims.
- **Bucket 2 (rare):**
  - 41% are cells never seen in training (new plan/CPT combinations).
  - The shared auths are mostly chemo/infusion drug + administration pairs in treatment courses.
  - **912 FN have a CPT whose overall PA rate is above the cut-off**, but the model didn't use it for the rare combination.
- **Bucket 3 (historically non-PA):**
  - 46% claim-level auth, 31% therapy.
  - Real change after training is concentrated in **NYCE PPO**, a new plan from Jan 2026: 594 FN. Its PT PA starts at visit 26, which is why first-PA months peak in **April 2026**.
  - No sign of a feed or label change across plans.
- **Bucket 4 (mixed):**
  - 53% claim-level auth.
  - 36% code denials on routine care: at Emblem HIP, denied lines are billed by someone other than the patient's PCP; at MVP and Anthem, specific sites dominate.
  - 6% NYCE PT; 5% own auth.
- **Bucket 5 (model miss):** all 92 are small cells (20–55 lines) that are 50–100% PA in training but scored ~0.12. The model's `min_data_in_leaf = 100` blends them with large non-PA cells.

### 3.3 Supporting evidence for the claim-level auth problem
TRUE rate when a code is **alone on the claim** vs **with other services** (2026):

| Code | Alone | With others | Read |
|---|---|---|---|
| G2211, 93000 (ECG), 36415 (blood draw) | 0.1–0.3% | 0.8–1.4% | Almost never PA on their own |
| 99213 / 99214 / 99215 | 0.5–1.2% | 1.2–3.5% | Rarely PA on their own |
| 93306 (echo), 93015 (stress test) | 10–12% | 13–31% | Genuinely need PA |

---

## 4. Top 3 experiments

### Experiment 1 – Take quality codes out of scope
**Idea.** Quality codes (CPT Category II, four digits + F, e.g. 3074F "systolic BP < 130") are $0 reporting tags. **No payer requires PA for them.** They are labelled PA only because they sit on an authorised claim. Scoring the model on them penalises it for something it should never predict.

**What we do.** Exclude `^\d{4}F$` codes from training and from evaluation.

**Expected effect.** About **1,930 FN** disappear from bucket 1. Precision also changes, because their FP/TP go too. The model stops spending capacity on codes that can never be PA.

**How we measure.** FN / FP / AUC-PR at 80% and 90% recall, service and visit level. Reported on **both** the full test set and the test set without quality codes, so the gain isn't just "we removed the hard cases".

**Why first.** Zero ambiguity, a one-line filter, no label debate.

---

### Experiment 2 – Let the model fall back on the CPT for rare and new cells
**Idea.** The model only knows exact combinations of CPT + modifier + POS + payor + plan. When a combination is rare or new (a new plan, a plan-name variant), it has almost no history and the score drops, even when the CPT itself is usually PA. Separately, the leaf minimum of 100 lines blends small all-PA cells with large non-PA ones.

**What we do (two settings, tested together and separately):**
1. Add **smoothed historical PA rates** at coarser levels as features: CPT, and CPT + payor. These are computed on the training window only (time-respecting, no test leakage). A rare combination then inherits what we know about its CPT.
2. Lower `min_data_in_leaf` from 100 to ~20.

**Evidence.**
- 912 rare-cell FN have a CPT whose overall PA rate is above the cut-off.
- All 92 bucket-5 FN are small cells with 50–100% PA, scored ~0.12.

**How we measure.** Same test labels as the benchmark. FN/FP at 80/90% recall, AUC-PR, plus FN recovered in buckets 2 and 5. Watch FP for over-fitting small cells.

**Why second.** It improves the model without any label decision; it's purely a modelling change.

---

### Experiment 3 – Don't count claim-carried auths on codes that don't need PA themselves
**Idea.** The auth number is written on the whole claim, so an office visit, blood draw or ECG next to an authorised echo, chemo or Botox injection is labelled PA. Our data shows these codes are **almost never PA when alone on a claim** (≤ 1%). Their PA label is inherited, not their own, and the model can't predict an inherited label from the service's own features.

**What we do.**
1. From training data, compute each CPT's **"alone on claim" PA rate**.
2. Codes below a threshold (e.g. < 2%: office visits, G2211, labs, ECG) become **"rarely own-PA" codes**.
3. Their PA label counts **only if the auth isn't shared** with another service on the claim; otherwise it's set to not-PA.
4. Retrain and compare.
5. Report **visit-level** metrics alongside. The visit-level label doesn't change under this rule, so it's a stable anchor.

**Expected effect.** Most of the remaining ~3,300 claim-level FN (beyond quality codes) leave the FN list. FP should also fall, because the model stops learning that office visits are sometimes PA for reasons it can't see.

**How we measure.** Because this changes labels, report on **old and new labels**, and at visit level (unchanged labels).

**Decision needed from the team.** Is the target "this service needs PA" (then this experiment is the right definition) or "this service appeared under an auth"? This is the main discussion point.

---

## 5. Parked (not in the top 3, and why)
- **Pre-visit context** (department, visit type, provider vs PCP): real signals, but they raise denial rates only to ~2% on most routine care. Worth adding after experiments 1–3.
- **History features** (prior denial, auth already open, therapy visit count): need confirmation that this history is available at prediction time.
- **NYCE PPO / Cigna therapy rules:** driven by visit counts across providers; recorded as "data not available at prediction time".
- **Code denials (197) on routine care:** a definition question (administrative vs clinical PA), decided together with experiment 3.

## 6. Caveats
- 74% of FN analysed. Bucket 2 coverage is 50% (cell limit), the others ~80%.
- Label-rule and claim-level checks used the first ~2M lines of 2026, not a random sample.
- Cigna therapy rule not yet verified (visit-number check and policy lookup pending).
- FN counts are measured on labels with the same biases; label experiments will move them.

---

## Appendix – reproduce the problem split for any bucket
```python
# after: python fn_root_cause.py --buckets <n> --cover 0.9 --out rc_b<n>
import numpy as np, pandas as pd
THERAPY = r"^(97\d{3}|9894[0-2]|92507|92526)$"
b = pd.read_csv("rc_b1/fn_causes.csv", dtype=str)
b["shared"] = b.t2_shared_with.fillna("").ne("")
code_lab = b.t1_source.isin(["PA denial code", "other remit row"])
b["problem"] = np.select(
    [code_lab,
     b.t1_source.eq("auth number") & b.shared & b.FINAL_CPT.str.match(THERAPY),
     b.t1_source.eq("auth number") & b.shared,
     b.t1_source.eq("auth number") & ~b.shared],
    ["code denial", "therapy", "claim-level auth", "own auth"], "no TRUE line found")
print(b.problem.value_counts())
print("quality codes:", int(b.FINAL_CPT.str.fullmatch(r"\d{4}F").sum()))
```

```python
# bucket 2 / 5 quick checks (rare cells and model setting)
fb = pd.read_csv("exp_nofilter_2022/fn_buckets_recall80_min20.csv", dtype={"FINAL_CPT": str})
r2, r5 = fb[fb.bucket.str.startswith("2")], fb[fb.bucket.str.startswith("5")]
print("rare: never seen", int((r2.train_n == 0).sum()), "| CPT rate >= cut-off", int((r2.train_cpt_pa_rate >= 0.421).sum()))
print("model miss:", len(r5), "| cells < 100 lines", int((r5.train_n < 100).sum()), "| median score", r5.score.median().round(3))
```
