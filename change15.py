1. Reflecting on the goals you had for this year, what were your key accomplishments?

Since joining in September, I’ve led the data science work on the Prior Auth Determination model (Delivery Excellence), from data preparation and label design through to evaluation and documentation.

Improved label quality, which measurably improved the model. I found two sources of label noise in the 27.5M-line claims dataset: quality-reporting codes, and “passenger” services that inherit another service’s auth number. I designed and built rules for both. On services where every label definition agrees, the passenger fix reduced false alarms at 80% PA catch by about 10% (40,880 → 36,633) and raised the share of real PA caught from 84.1% to 86.0%, consistently across two random seeds.
Built a rigorous, repeatable evaluation framework: a frozen test set, a model-by-answer-key comparison so label changes can’t flatter results, seed checks, a validation-based cut-off check, and a model registry with traceable run IDs. This let the team compare experiments fairly and see real progress rather than artefacts of changing labels.
Made a data-driven case for new features. A precision-ceiling analysis showed that 90% precision isn’t reachable with the current five fields: even a perfect model on them reaches about 86% at 80% catch, and a realistic one about 66%, against our 51%. This set clear expectations with stakeholders and pointed the roadmap at new information (diagnosis, provider, prior-auth history) rather than more tuning.
Brought governance and transparency to label decisions. For the passenger rule I produced an evidence analysis per CPT and payer. It showed that every relabelled line had shared another service’s auth, and it surfaced the cases where a payer’s own data disagreed. I presented options to the team, implemented the option they chose (option C), and tested it. The test showed the original passenger labels still train the better model, so I recommended training on those and reporting on the more cautious definition.
Gave the team reusable tools: false-negative bucket analysis, a precision-ceiling check and a per-CPT lookup, all documented, so analysis can be rerun after every training run.
Assessed a new data source: I profiled the AccuReg HL7 registration data. It has diagnoses on about 97% of messages and pre-visit auth status, but only about 3% of its payer names match our claims. I raised the linkage and crosswalk questions needed before it can be used.
Documented the work: an experiment progress report and a passenger-label analysis for the team, with assumptions and open risks stated explicitly.

2. What challenges did you face and how did you develop?

Ramping up on a new domain quickly. I joined mid-year, new to claims data, CPT/CARC coding, prior-auth workflows and HL7. I learned by profiling the data directly and checking assumptions against standards and the data, rather than relying on intuition.
No ground truth for labels. We can’t see which services an auth was actually approved for, so label rules rest on indirect evidence. I learned to state clearly what’s measured and what’s inferred, to quantify the remaining risk, and to identify what data would settle it (auth approval records).
Communicating complex results simply. Early results could be read several ways: different cut-offs, answer keys and metrics. When stakeholders found that confusing, I simplified to the view that matches the business priority, missed PA at equal review workload, and kept the other views as supporting checks.
Accepting results that didn’t go our way. The team’s option C labels didn’t improve the model. I reported that plainly, with the evidence, and proposed a way forward that still met the team’s concern.
Working within data and tooling constraints: data access policies, large S3 datasets and limited features. I built lightweight, self-contained scripts so the team could reproduce the analysis without depending on my environment.

3. What do you want to continue to work towards in the coming year?

Production readiness: set the decision cut-off on recent data, add monthly refresh and drift monitoring (the PA rate moved from about 5.8% in training to 3.1% in test), and agree the operating point with the business.
New predictive information: confirm which fields the API can supply at prediction time, and test diagnosis, provider and prior-auth-history features. If a link key exists, include AccuReg registration data.
Recent-data training: follow up an early single-seed result where training on 2025 onwards cut false alarms by about 17%, and confirm it across seeds before adopting it.
Governance and responsible AI: take the model through AIRB with documented bias, fairness and explainability artefacts. Keep protected attributes out of features. Close the open label questions: the referral / visit-level-auth decision, and sign-off of the passenger rule.
Technical leadership: turn the evaluation tools into a standard part of every training run, and help the team adopt them. [Add any personal development goals, e.g. MLOps depth, leading design reviews.]

Q1: replace the [Technical Leadership] placeholder with:

Shaped the project’s technical direction early. Although I joined in September, I quickly became one of the main drivers of the modelling approach. I moved the team from tuning the model to fixing the labels and measuring what the data can support. I introduced the frozen test set, the label-independent comparison and the precision-ceiling analysis, which are now the basis of how results are judged.
Recommended alternative solutions backed by evidence. For example:
I showed that 90% precision isn’t reachable with the current fields. That redirected effort to new data sources instead of more tuning.
When the team’s preferred label option (option C) didn’t improve the model, I proposed training on the original passenger labels and reporting on the more cautious definition. I suggested a business rule on top of the model if those cases must be caught.
I recommended parking time-based validation and skipping a further label build when the evidence showed low return. That saved compute and team time.
I advised against building payer mappings from assumptions. A governed crosswalk is needed first.
Supported and reviewed others’ work. I reviewed a contractor’s work, checking his approach, results and conclusions, and gave feedback to make it rigorous and comparable with our evaluation standard. [Add one concrete example if you can, e.g. a leakage issue, a wrong metric or a missing baseline you spotted.]
Guided technical decisions with the project lead. Where the right approach was unclear, I helped set the direction by laying out options with evidence and trade-offs. Examples are the passenger-label options A–F and how to treat visit-level auth. This let the team make informed decisions quickly.

Q2: add one challenge:

Leading without formal authority as a new joiner. I was new to the organisation and the domain, so I couldn’t rely on history or position to influence decisions. I earned trust by backing every recommendation with data, being open about what was measured and what was inferred, and reporting results honestly even when they didn’t support the preferred option.

Q3: replace the Technical Leadership bullet with:

Technical leadership:
Keep driving the technical direction of the PA model through to production.
Make the evaluation framework a standard part of every training run.
Formalise reviews of contractor and team work against it.
Mentor teammates on ML evaluation and label-quality methods.
Develop my own depth in MLOps and deployment so I can lead the move from experiment to production.
















Q1: add under Delivery Excellence (after the evaluation framework bullet):

Designed a way to update the model quickly, without full retraining. Payer PA rules change during the year, and retraining the full model each time is slow and needs re-validation. I built a cell-weight layer on top of a frozen LightGBM model:
How it works: for each CPT + plan combination (and a finer CPT + modifier + place of service + plan level), it learns a correction from the latest three months of data. The correction is the gap between the observed and predicted PA rate. Cells with little data are shrunk towards zero so they don’t swing on a few lines.
Isolation: a weight changes only its own cell, and the rest of the model stays exactly the same. I proved this with an isolation test that edited 5 cells and saw zero change everywhere else. This makes every update targeted and auditable.
Known rule changes: when a payer announces a change, the cell can be set directly (e.g. to “PA required”) from day one, before any claims arrive.
Results:
On real recent data, learned weights closed [X%] of the gap between the frozen model and a full retrain.
In a simulation where cells were made to newly need PA, or stop needing it, [manual / learned] weights correctly re-scored [X%] of the changed lines, with no change to any other cell.
Built a base + adapter design so new features (e.g. payer rules coverage, diagnosis) can be added as a small model on top of the frozen base. That way the effect of a new feature can be measured, and turned off, without touching the base model.

Q3: add to the Production readiness bullet:

…and move the cell-weight update into production as the routine way to absorb payer rule changes between full retrains. Set the process for who approves a manual cell change, and how it is logged and reviewed.
