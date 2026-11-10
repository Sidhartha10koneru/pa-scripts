The v8 draft dataset is ready: s3://cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v8_draft/ 

It’s the std v4 data (2022 to Jul 2026, 27.5M lines, nothing dropped) with all label versions as separate columns, so we can train and compare any of them from one dataset. It’s a draft: the passenger rule hasn’t been tested in training or signed off yet.

Label columns

Column	What it does	TRUE lines
GROUP_LABEL	Original label	547,317
LABEL_V6	Quality codes (CPT II, ####F) → FALSE	526,292
LABEL_V7	v6 + any of the 17 Acuity auth-denial codes → TRUE	1,290,125
LABEL_V7_STRICT	v6 + only the 10 auth-only Acuity codes → TRUE	530,661
LABEL_V6_PAX / LABEL_V7_PAX / LABEL_V7_STRICT_PAX	Same, with “passenger” lines → FALSE (service TRUE only because another service’s auth number is on the same claim)	502,761 / 1,268,101 / 508,637

Key stats

Quality codes: 21,025 TRUE → FALSE.
Full Acuity list: +763,833 TRUE (336,680 from FALSE, 427,153 from AMBIGUOUS). About 99% of these come from codes 59, 16 and 96, which in X12 aren’t auth codes on their own.
Strict Acuity list: +4,369 TRUE.
Passengers: about 23.5k TRUE → FALSE, mostly office visits, G2211, labs and ECG. 7,120 visit-level auths kept TRUE pending the referral decision.
No line with an Acuity code is set FALSE in the v7-based labels.

Which column to use

Baseline: GROUP_LABEL, with quality codes excluded (PA_QUALITY_MODE=exclude).
Recommended candidate to test: LABEL_V7_STRICT_PAX.
Also testing: LABEL_V6_PAX (passenger effect only) and LABEL_V7_PAX (full Acuity list).

Note: there is no LABEL_V8 column. The v8 draft adds the passenger columns (*_PAX, IS_PASSENGER, VISIT_LEVEL_AUTH) on top of the v6 and v7 labels.”
