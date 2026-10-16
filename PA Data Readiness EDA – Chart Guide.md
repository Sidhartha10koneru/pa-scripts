# PA Data Readiness EDA – Chart Guide

Oct 16, 2026 · @Sid

## Key terms

Every chart from `build_eda_charts.py` (v4) counts one of two units and uses one of two bases for its percentages; this guide states both for each chart. Data only: no rules file or master sheet is used.

| Term | Meaning |
| --- | --- |
| Line | One row of the parquet = one billed code (CPT/HCPCS) on a claim |
| Encounter | One distinct CSN. All lines of a CSN are collapsed into one row and counted once. Average 3.3 lines per CSN |
| TRUE / FALSE / AMBIGUOUS | GROUP\_LABEL. TRUE = auth number or one of 7 CARC/RARC codes; FALSE = clean paid, no auth signal; AMBIGUOUS = conflicting or insufficient evidence |
| All lines | TRUE + FALSE + AMBIGUOUS |
| Labelled | TRUE + FALSE only. AMBIGUOUS is left out |
| PA line | A line with GROUP\_LABEL = TRUE |
| PA encounter | A CSN with at least one TRUE line (TRUE only) |
| Labelled encounter | A CSN with at least one TRUE or FALSE line |
| PA rate | PA (TRUE) ÷ labelled (TRUE + FALSE), in lines or encounters as stated |
| Cell | One combination of codes, e.g. CPT + payor + plan, or the PRD cell CPT + modifier + payor/plan + place of service |
| Flagged codes | A hand-picked, unverified list of codes that may ride along on a PA claim (visits, Category II, blood draw, infusion admin, contrast, G2211). Used only to flag, never to drop |

The label is set per claim and copied to every line (only 0.55% of claims mix TRUE and FALSE), so line counts overstate PA examples by about 2.6x. Encounter counts are the real number of PA examples.

## All charts at a glance

Seven charts drive the readiness verdict (01, 05, 07, 09, 12, 14, 15); the rest are diagnostics or context. All charts include every filing order (primary and secondary) and every year passed to the run.

| Chart | Unit counted | Labels used | Left out | % is of | Role |
| --- | --- | --- | --- | --- | --- |
| 00 Scorecard | Mixed (per row) | Per row | – | Per row | Summary |
| 01 Funnel | Lines | TRUE, FALSE, AMBIGUOUS | Nothing | All lines (usable %); labelled lines (PA rate) | Decision |
| 02 Payor rate | Lines | TRUE, FALSE | AMBIGUOUS; payors with < 100 labelled lines | Labelled lines per payor | Informational |
| 03 Feature association | Lines | TRUE, FALSE | AMBIGUOUS; empty values of each feature | Not a % (Cramér's V, 0–1) | Informational |
| 04 Monthly stability | Lines | TRUE, FALSE | AMBIGUOUS; lines without a service date | Labelled lines per month | Diagnostic |
| 05 PRD volume gates | Encounter × PRD cell | TRUE, FALSE | AMBIGUOUS | Cells; labelled encounters; PA encounters | Decision |
| 06 Cell purity | Encounter × PRD cell | TRUE, FALSE | AMBIGUOUS; cells with < 20 labelled encounters | Labelled encounters; PA encounters | Supporting |
| 07 Label by year | Left: lines. Right: encounters | Left: all three. Right: TRUE, FALSE | Right: AMBIGUOUS-only CSNs | Left: all lines. Right: labelled encounters | Decision |
| 08 Label noise | Lines | TRUE only | FALSE, AMBIGUOUS | TRUE lines | Informational |
| 09 Filing order | Lines | TRUE, FALSE | AMBIGUOUS | Labelled lines per filing order | Decision |
| 10 Drift | Lines | TRUE, FALSE (mix only, label ignored) | AMBIGUOUS | Not a % (PSI) | Diagnostic |
| 11 Missingness | Lines | Left: all three. Right: TRUE vs FALSE | Right: AMBIGUOUS | Left: all lines. Right: TRUE lines / FALSE lines | Diagnostic |
| 12 Claim structure | Lines and encounters | All three (mixed-label: TRUE, FALSE) | Nothing | Claims; all lines; labelled claims | Decision |
| 13 Coverage | Lines | TRUE, FALSE | AMBIGUOUS | Latest year's labelled lines / TRUE lines | Supporting |
| 14 Baseline | Encounter × PRD cell | TRUE, FALSE | AMBIGUOUS | Not a % (precision, recall, AP) | Decision |
| 15 CPT solo vs with others | Encounter × CPT | TRUE, FALSE | AMBIGUOUS | Labelled encounters containing the CPT | Decision (label fix) |
| 16 Field presence | Lines | TRUE, FALSE | AMBIGUOUS | Labelled lines filled / empty | Diagnostic (leakage) |
| 17 Line of business | Encounter × LOB | TRUE, FALSE | AMBIGUOUS; LOBs with < 500 labelled encounters | Labelled encounters per LOB | Informational + label check |

**Role:** Decision = feeds the readiness verdict. Supporting = backs a decision chart but is not proof on its own. Diagnostic = flags a data problem to investigate. Informational = context; do not draw conclusions from it alone.

## Charts 00–04

### 00 Readiness scorecard

- **Input:** the metrics computed by every other chart, one row per readiness question.
- **How to read:** each row shows the metric, its value, a status and the rule behind the status. PASS / WARN / FAIL come from thresholds; INFO rows have no threshold.
- **What it says:** a one-page verdict. The thresholds are our starting points, except the PRD numbers (100, 500, floor 50); agree them before presenting.
- **Known false alarms:** "Features complete" and "Missingness unrelated to label" fail because of FINAL\_MODIFIERS, which is legitimately empty on most lines. "Optional fields leak" shows a ratio that explodes when the empty group has almost no PA (see 16).
- **Role:** summary.

### 01 Funnel

- **Input:** all lines, counted by GROUP\_LABEL (TRUE, FALSE, AMBIGUOUS). Nothing left out.
- **How to read:** bars on a log scale: all lines → usable (TRUE + FALSE) → TRUE → FALSE. The title gives usable lines as a % of all lines.
- **What it says:** how much of the data carries a usable label. About 53% of lines are usable; about 45% are AMBIGUOUS and unusable for training.
- **Role:** decision (label coverage).

### 02 Payor rate

- **Input:** labelled lines (TRUE + FALSE) per payor; payors with fewer than 100 labelled lines left out.
- **How to read:** the 10 payors with the highest PA rate (TRUE lines ÷ labelled lines). Dashed line = overall PA rate across all payors.
- **What it says:** which payors have the highest share of PA lines. It ranks by rate, so small payors dominate and it shows no counts. Very high rates (MetroPlus 79–94%, American Specialty Health 96%) are worth a label check.
- **Role:** informational. For PA counts by payor and plan use `pa_by_payor_plan.xlsx`.

### 03 Feature association

- **Input:** labelled lines; for each feature, a TRUE/FALSE count per value. Empty values are left out by the crosstab.
- **How to read:** bias-corrected Cramér's V from 0 (no link to PA) to 1 (perfect link). The number in brackets is how many distinct values the feature has. The correction stops features with thousands of values (NPI, CPT, diagnosis) from looking strong by default.
- **What it says:** which single fields carry the most signal. Latest run: CPT 0.62, department 0.60, visit type 0.59, NPI 0.46, modifier 0.45, diagnosis 0.38; place of service only 0.07.
- **Caveats:** one feature at a time, in-sample, line level. High values for department and NPI can mean the model memorises sites or providers rather than learning rules.
- **Role:** informational.

### 04 Monthly stability

- **Input:** labelled lines grouped by service month; lines without a valid service date left out.
- **How to read:** grey bars = labelled lines per month (left axis); red line = PA rate per month (right axis). The scorecard metric (max ÷ min monthly rate) uses only months with at least 1,000 labelled lines.
- **What it says:** whether the PA rate is steady over time or jumps. Jumps point to changes in data feeds or label construction; a drop in the most recent months can mean claims are not yet fully adjudicated.
- **Role:** diagnostic.

## Charts 05–07

### 05 PRD 7.3 volume gates

- **Input:** labelled lines reduced to one row per encounter per PRD cell (CPT + modifier + payor + plan + place of service), then TRUE and FALSE encounters counted per cell. AMBIGUOUS left out. A blank modifier or POS is its own value "NONE".
- **How to read:** each cell is put in a PRD tier: rules only (< 100 labelled encounters), model may say "required" only (100–500 and at least 50 TRUE), model both directions (> 500 with at least 50 TRUE and 50 FALSE), or below the 50-example floor. Three bars per tier: % of cells (grey), % of labelled encounters (blue), % of PA encounters (red).
- **What it says:** how much PA the model is allowed to answer under the PRD rules. Latest run: the model may say "required" for 51% of PA encounters; 43% sit in rules-only cells. The large "not required" share is driven by Category II quality codes (3008F, 3074F, 3078F), which trivially need no PA.
- **Caveats:** one encounter with two CPTs counts in two cells. Counts cover all years in the run; whether the PRD means per year is not confirmed. `05_prd_cells.csv` lists every cell with its tier.
- **Role:** decision.

### 06 Cell purity

- **Input:** the same encounter × PRD cell counts as 05; only cells with at least 20 labelled encounters.
- **How to read:** cells grouped by their PA rate (< 10%, 10–50%, 50–90%, ≥ 90%). Blue = % of labelled encounters in each group; red = % of PA encounters. The note under each red bar repeats the figure excluding flagged codes.
- **What it says:** whether PA cases sit in cells where PA is the norm. Latest run: 79% of PA encounters are in majority-PA cells.
- **Caveats:** in-sample and partly built in, because the label rule moves contradictory FALSE records to AMBIGUOUS. Not proof of learnability; chart 14 is.
- **Role:** supporting.

### 07 Label by year

- **Input, left panel:** all lines per year, split TRUE / FALSE / AMBIGUOUS. The line is the PA rate in lines (TRUE ÷ labelled lines).
- **Input, right panel:** distinct CSNs per year. Red bar = PA encounters (TRUE only). Orange bar = PA encounters with a TRUE line outside the flagged list. The line is the encounter PA rate (PA encounters ÷ labelled encounters).
- **How to read:** left shows how the label mix changes by year; right shows the real number of PA examples per year.
- **What it says:** latest run: PA encounters 27k (2022), 28k, 53k, 91k, 54k (2026, part year); encounter PA rate 4.6%, 5.0%, 7.0%, 5.0%, 3.2%. Volume is not the constraint. The 2026 dip may be claims not yet fully adjudicated.
- **Role:** decision (volume and stability of the label).

## Charts 08–12

### 08 Label noise (flagged codes)

- **Input:** TRUE lines only, per year. FALSE and AMBIGUOUS left out.
- **How to read:** % of TRUE lines whose CPT is on the hand-picked flagged list.
- **What it says:** 22–34% of TRUE lines are on flagged codes, rising to 34% in 2026. The list is unverified; chart 15 tests each code with data and shows the list was partly wrong (office visits do not ride along).
- **Role:** informational; superseded by 15.

### 09 Filing order

- **Input:** labelled lines grouped by FILING\_ORDER\_C (1 = primary, 2 = secondary, …; -1 = missing). AMBIGUOUS left out of the rate.
- **How to read:** bar = PA rate (TRUE ÷ labelled lines) per filing order; the note gives that order's share of labelled lines.
- **What it says:** primary lines 3.85% PA (92.5% of lines); secondary 2.50% (7.3%). Secondary lines still carry TRUE, because the auth on the primary claim is copied to the secondary claim of the same encounter (same CSN).
- **Role:** decision (train on primary, filing order 1).

### 10 Drift

- **Input:** labelled lines; the mix of payors and of CPTs per year. The label itself is not used.
- **How to read:** PSI (population stability index) of each year against the latest year, on the top 30 values plus OTHER. Below 0.1 = stable; 0.1–0.25 = moderate shift; above 0.25 = major shift.
- **What it says:** how different older years look from the newest. A high PSI can come from real mix change or from payor names changing between years; check which values drive it before deciding which years to train on.
- **Role:** diagnostic.

### 11 Missingness

- **Input, left:** all lines per year (AMBIGUOUS included). **Right:** TRUE lines vs FALSE lines, all years. Placeholder values ("nan", "UNKNOWN", "", "0") count as empty.
- **How to read:** left = % of lines with the field empty, per year; right = % empty among TRUE lines vs among FALSE lines.
- **What it says:** core fields are about 0% empty. FINAL\_MODIFIERS is 65–82% empty, which is normal (most lines have no modifier). POS, LOB and product are 2–7.6% empty, rising in 2026; TRUE lines almost never have them empty.
- **Role:** diagnostic. The scorecard rows that fail on this chart are driven by modifiers.

### 12 Claim structure

- **Input:** all lines and all CSNs; the mixed-label figure uses CSNs with labelled lines.
- **How to read:** bars = % of claims (CSNs) by number of lines (20 = 20 or more). Title: average lines per claim; repeated lines (same CSN + CPT + date) as % of all lines, with and without modifier + POS; % of labelled claims that mix TRUE and FALSE.
- **What it says:** 3.3 lines per claim; 16.4% of lines repeat even with modifier + POS (units or resubmissions, to check); only 0.55% of claims mix labels, so the label is set per claim. Train/test must be split by CSN, never by line.
- **Role:** decision (how to split and de-duplicate).

## Charts 13–17

### 13 Coverage

- **Input:** labelled lines of the latest year, compared with labelled lines of all earlier years in the run.
- **How to read:** for CPT, payor and CPT + payor + plan: % of the latest year's labelled lines (blue) and TRUE lines (red) whose value was seen in earlier years with at least 20 labelled lines.
- **What it says:** 98% of 2026 PA lines have a CPT and 93% a payor that earlier years saw well, but only 58% an exact CPT + payor + plan combination. A model learning from separate features can generalise; a lookup of exact combinations cannot.
- **Caveat:** lines, not encounters.
- **Role:** supporting.

### 14 Baseline (learnability over time)

- **Input:** encounter × PRD cell counts (TRUE, FALSE). Earlier years = training; latest year = test.
- **How to read:** left = precision–recall curve of a lookup that predicts each test encounter's PA rate from earlier years (PRD cell, falling back to CPT + payor + plan, CPT + payor, CPT, then the overall rate). Dashed line = base rate. AP (average precision) summarises the curve; AP ÷ base rate is the lift. Right = which lookup level answered.
- **What it says:** train 2022–25, predict 2026: AP 0.43 vs base rate 0.027, about 16x. The signal is real and holds over time; any model must beat AP 0.43. Excluding flagged codes, AP is 0.53.
- **Caveat:** a lookup table, not a model; it is the floor to beat.
- **Role:** decision (the evidence for learnability).

### 15 CPT alone vs with other codes

- **Input:** labelled lines reduced to one row per encounter per CPT. An encounter is "solo" for a CPT when that CPT is its only distinct code, otherwise "with others". AMBIGUOUS left out.
- **How to read:** the top 25 CPTs by PA encounters (TRUE only), chosen by data, not by hand. Blue = PA rate when the CPT is alone; red = PA rate when other codes are present. \* = also on the flagged list (does not affect selection). Label: rides along = solo rate below 20% of the with-others rate, with at least 100 solo encounters and 20 PA encounters.
- **What it says:** contrast (A9575, Q9967), chemo administration (96413) and 97112 ride along: their TRUE comes from another code on the claim. MRIs carry PA themselves. Office visits have low PA either way. Ride-along CPTs touch 29.7% of PA encounters.
- **Caveats:** one encounter counts once for every CPT on it, so counts overlap. The 20% rule is ours. Full list for every CPT: `15_cpt_solo_vs_with.csv`.
- **Role:** decision (candidate list for the label fix, for business review).

### 16 Field presence (leakage check)

- **Input:** labelled lines, split per field into filled and empty.
- **How to read:** red = PA rate among lines with the field filled; grey = PA rate among lines with it empty. "filled X%" = share of all labelled lines with a value. Ratio = red ÷ grey.
- **What it says:** POS, LOB and product are filled on 93% of lines and on almost every PA line; the 7% without them are almost never PA (about 0.003%), which makes the ratio look huge. Modifier (2.3x) and diagnosis (1.9x) gaps make clinical sense.
- **Caveat:** quote the fact, not the ratio. Ask the data team which feed the unenriched 7% come from.
- **Role:** diagnostic (leakage).

### 17 Line of business

- **Input:** labelled lines reduced to one row per encounter per LOB; LOBs with fewer than 500 labelled encounters left out. AMBIGUOUS left out.
- **How to read:** bar = PA rate (PA encounters ÷ labelled encounters) in that LOB; n = labelled encounters. Dashed line = overall rate (4.62%). Orange = LOBs where health-plan PA is not expected.
- **What it says:** highest rates among real segments: Medicaid Replacement (Medicaid managed care) 9.2%, Medicare Replacement (Medicare Advantage) 7.6%, Medicaid 7.5%. Commercial is 3.4% but supplies the most PA examples (about 42%). Original Medicare 0.09%. Self-pay 7.1% and workers' comp 31.7% point to label issues, not payer PA.
- **Caveats:** LOB may be the account's financial class; an encounter under two LOBs counts in both.
- **Role:** informational + label check.

## Other outputs and caveats

| File | What it holds |
| --- | --- |
| `00_readiness_scorecard.csv` | The scorecard as a table |
| `05_prd_cells.csv` | Every PRD cell: TRUE and FALSE encounters, total, tier, PA rate |
| `15_cpt_solo_vs_with.csv` | Every CPT: solo and with-others counts and rates, pattern, on flagged list |
| `readiness_metrics.json` | Every number behind every chart |
| `REMARKS.md` | Auto-written observations per chart, with a blank "Team remark" line. Overwritten on each run: edit a copy |
| `pa_by_payor_plan.xlsx` (separate cell) | PA encounters, PA rate and share of all PA per payor + plan, all years and per year |

**Caveats that apply to every chart**

- **All filing orders are included.** Secondary-payer lines (7.3% of labelled lines) are in every count; restricting to filing order 1 would lower the counts slightly.
- **The years are those passed to the run.** The all-years run and the 2024–26 run give different numbers; say which one a figure comes from.
- **Thresholds are ours,** except the PRD numbers (100, 500, floor 50): 20 labelled for a reliable rate, 5 PA, 500 per LOB, the 20% ride-along rule, and the scorecard PASS/WARN/FAIL bands.
- **CSN is treated as the encounter.** Primary and secondary claims of one visit share a CSN; this is inferred from the data and should be confirmed by the data team.
- **The flagged-code list is unverified.** Nothing is dropped because of it; chart 15 is the data-based view.
