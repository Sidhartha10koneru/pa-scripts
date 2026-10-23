This plan is built on what we already have. Every accuracy number is either a measured baseline from our runs, or a decision threshold I’m proposing. I haven’t predicted any results; those come from the runs.

Shared test setup (used by all tasks)
Window	Data	Use
Train	2025	base model
Update	Jan–Apr 2026	refit, cell weights, recency
Test	May–Sep 2026	final scoring; never used for updating

Measured so far (10% sample, test = all 2026, so these will shift once the window changes):

Encounter AP: 0.626 base; 0.650 with rules.
PA caught at 90% precision: 3.5% base; 9.5% rules adapter; 25.7% full retrain with rules.
Auto-clear at 99.5% confidence: 87% of encounters, with 14% of PA wrongly cleared.

Metrics for every task:

encounter AP;
PA caught at 90% / 94% precision;
auto-clear share and PA wrongly cleared at 99.5%;
calibration error (do scores match real rates?);
time to update;
isolation (did anything else change?).
Task 1: cell-level weights (frozen model)
Sub-task	Model / method	Day	Output (milestone)	Success measure
1.1 Find changed cells	data analysis: cells (CPT + plan, n ≥ 50) whose PA rate moved most from 2025 to Jan–Apr 2026	D2	list of the top 20 changed cells	at least 10 cells with clear change
1.2 Build the weight table	LightGBM base + weight table (key, weight, effective date, expiry, reason)	D2	table and scoring function	works end to end
1.3 Isolation test	set weights on 5 cells	D2	before/after comparison	0 changes outside edited cells
1.4 Manual weights	formula: force / target probability	D3	scores on the test window	edited cells reach their target probability
1.5 Learned weights	offsets learned from the update window, shrunk for small cells (k = 20 / 50 / 100)	D3	comparison table	changed cells’ error falls; no loss overall
1.6 EBM (optional)	Explainable Boosting Machine, editable CPT × plan table	D6	AP vs LightGBM, edit test	within 0.02 AP of LightGBM
Task 2: refit leaf values
Sub-task	Model / method	Day	Output (milestone)	Success measure
2.1 Refit function	LightGBM refit() on the update window	D4	script option --refit	runs; trees unchanged
2.2 Decay sweep	keep 30 / 50 / 70 / 90% of old values	D4	metrics per setting	best setting chosen
2.3 Compare	refit vs frozen vs recency adapter vs full retrain	D4	comparison table	refit AP ≥ frozen + 0.01, or calibration error ≥ 30% lower; within 0.01 of a full retrain
2.4 Cost	time to refit	D4	minutes	under 10 min on the full update window
Task 3: time-decay weights
Sub-task	Model / method	Day	Output (milestone)	Success measure
3.1 Weighting	LightGBM full retrain on 2024 → Apr 2026, weight halves every 3 / 6 / 12 months	D5	script option --half-life	runs
3.2 Unweighted control	same data, no weights	D5	control metrics	n/a
3.3 Compare	best half-life vs unweighted vs refit vs frozen	D5	comparison table	weighted AP ≥ unweighted + 0.01 to justify it
3.4 Answer	how much history to keep, how often to retrain	D5	one-paragraph conclusion	clear recommendation
Day-by-day schedule
Day	Date	Work	Milestone
D1	Mon 26 Oct	shared split; frozen base and full retrain baselines; add calibration error to the script	baseline table with all metrics
D2	Tue 27 Oct	1.1–1.3	changed-cells list; isolation proven
D3	Wed 28 Oct	1.4–1.5	cell-weight results
D4	Thu 29 Oct	2.1–2.4	refit results
D5	Fri 30 Oct	3.1–3.4	time-decay results
D6	Mon 2 Nov	combined (refit + cell weights); EBM optional; bigger sample on the best options	combined results
D7	Tue 3 Nov	decision table and write-up	team review and decision
Decision table (filled at D7)
Method	Encounter AP	PA caught @90%	Auto-clear @99.5% / PA missed	Calibration error	Update time	Isolated?	Decision
Frozen base					n/a	n/a	reference
Cell weights (manual)					minutes	yes	
Cell weights (learned)						yes	
Refit (best setting)						no	
Recency adapter						no	
Time-decay retrain						no	
Full retrain						no	upper bound

Proposed decision rules for the team to agree on before D1:

Cell weights: adopt if isolation is exact and the changed cells improve with no overall loss.
Refit: adopt if it beats frozen by at least 0.01 AP or cuts calibration error by at least 30%, and lands within 0.01 of a full retrain.
Time decay: adopt if it beats unweighted by at least 0.01 AP. Otherwise use plain retrains.
Risks to note in the sheet
Small numbers at 90% precision (a few hundred encounters): use a bigger sample on D6 for the finalists.
The 2025 → 2026 PA rate drop (4.25% → 2.9%) may be a labelling change rather than real behaviour. Confirm with Raghu before reading too much into it.
The data ends around Sep 2026, so check that the test window has enough months.

I can make D1 ready now: the split options and calibration error in the script. The sub-task tables are ready to paste into your Loop list.

we need 6 days for each task?

No. The 7 days cover all three tasks together, not 6 days each.

	Days	Which days
Shared setup (baselines, test split, calibration metric)	1	D1, used by all tasks
Task 1: cell weights	2 (+1 optional for EBM)	D2–D3 (+D6)
Task 2: refit	1	D4
Task 3: time decay	1	D5
Combine the best options + bigger sample check	1	D6
Decision table + write-up	1	D7
Total	7 working days	26 Oct – 3 Nov

For the task sheet, estimate each task at its own days plus a share of D1, D6 and D7:

Task 1: about 2–3 days.
Task 2: about 1–1.5 days.
Task 3: about 1–1.5 days.

The schedule assumes you’re doing all three, one after another. If a teammate takes Task 3, it can run in parallel with Task 1 once D1 is done, and the whole plan shrinks to about 5 days.

all three are for me

Then the 7-day schedule stands. It’s already sequential, built for one person:

Day	Date	Work
D1	Mon 26 Oct	shared setup and baselines
D2–D3	Tue 27 – Wed 28 Oct	Task 1: cell weights
D4	Thu 29 Oct	Task 2: refit
D5	Fri 30 Oct	Task 3: time decay
D6	Mon 2 Nov	combine the best options + bigger sample (EBM if time allows)
D7	Tue 3 Nov	decision table + team review

If you need it shorter (5 days):

Drop the EBM.
Do Task 2 and Task 3 on the same day; both reuse the same data load and test setup.
Fold the bigger-sample check into the write-up day.

You lose the EBM comparison and some confidence in the small high-precision numbers.

To keep it on track:

I’ll add the script options before you start: test-window split, calibration error, --refit, --half-life, and the cell-weight table. Then each day is mostly running and reading results, not coding.
If one task overruns, Task 1’s learned weights (1.5) and the EBM are the parts to cut first.

for below we might have it or not, but we should be able to adapt when new change comes in

1.1 Find changed cells    data analysis: cells (CPT + plan, n ≥ 50) whose PA rate moved most from 2025 to Jan–Apr 2026    D2    list of the top 20 changed cells    at least 10 cells with clear change

Working out how to detect or simulate cell changes in data.

Agreed. The task should prove we can adapt when a change arrives, whether or not the history happens to contain one. So 1.1 becomes “get change scenarios”, from three sources, with a controlled simulation as the guaranteed fallback.

Revised sub-task 1.1
Sub-task	Method	Day	Output	Success measure
1.1a Real changes in data (if any)	cells (CPT + plan, n ≥ 50) whose PA rate moved clearly from 2025 to Jan–Apr 2026	D2	list of changed cells; may be short or empty	n/a: a bonus if found
1.1b Real rule-list changes	compare the old REQ Excel with the new Prod_Globals: codes added or removed per payor	D2	list of added/removed codes per payor, with claim volume	a realistic set of “payer changed its list” cases
1.1c Simulated changes (always done)	pick about 20 cells of different sizes and inject a change from an effective date in the update + test windows: (i) newly requires PA (labels → TRUE), (ii) no longer requires PA (labels → FALSE)	D2	controlled test set with known changes	the change is known exactly, so adaptation can be measured precisely

Why simulation is the core: we decide what changed and when, so we can measure exactly how well each method recovers. It works whether or not real changes exist. Real changes (1.1a / 1.1b) then confirm it on actual cases.

Cases to cover:

both directions, with “newly requires PA” as the risky one;
small and large cells;
a cell the model was very sure about (hardest to flip) and one it was unsure about.
What “adapting” means: measured for each change
Measure	Meaning
Error on changed cells, before vs after	the frozen model is wrong after the change; does the cell weight fix it?
Time to correct	cell weight: immediately from the effective date. Refit / time decay: only once new labelled claims arrive.
Spillover	do predictions for unchanged cells move? Cell weights: should be 0.
Wrong decisions avoided	how many encounters would have been denied or over-flagged in the test window without the fix

This also compares all three tasks on the same change scenarios:

cell weights react immediately if we know about the change;
refit and time decay react only after data arrives.

That difference is exactly the point you raised earlier, and now it’s measured rather than argued.

Updated success measure for Task 1

Given a known change (simulated or real), a cell weight corrects the affected cells from the effective date, with zero spillover to other cells and no retraining.

How is Claude doing this session?


