# Quality Codes (CPT Category II) – Label Analysis

Nov 4, 2026 · @Sid

## Summary

Quality codes (CPT Category II, four digits + F, e.g. 3074F) are reporting tags that never need PA; about 21,000 of our PA labels on them are inherited from another service's auth on the same claim, so we take them out of the PA definition.

- **What they are:** $0 performance-measurement codes (blood pressure, BMI, medication list). No payment, so no prior authorisation applies.
- **Why they are labelled PA:** the auth number is written on the whole claim, so every line on an authorised claim becomes TRUE. 99.5% of quality-code TRUE lines share their auth with another service.
- **Size:** 84 distinct codes, 6.58M lines (24% of all lines), 21,025 TRUE lines (3.8% of all TRUE) across 2022–2026.
- **Effect on PA:** 19,263 PA services (3.9%) change; only **43 PA visits** across five years lose their PA label.
- **Risk of losing genuine PA:** low. 4 quality-code TRUE lines exist alone on a claim out of 6.4M; open spot checks are listed under Risks.
- **Decision:** v6.0 labels quality codes FALSE (`LABEL_V6`), flags them (`IS_QUALITY_CODE`), and keeps the original `GROUP_LABEL`. The model comparison favoured excluding them from training (separate write-up).

## How we found them

They surfaced in the label audit, were set aside when bucketing the model's misses, and were confirmed with auth numbers on the test misses.

1. **Label audit.** Office visits showed 65–96% PA for some plans in 2025, which is implausible. Their TRUE labels came from real auth numbers, so we checked whose auth it was. Comparing auth numbers across the services of one visit, the largest group sharing another service's auth was quality codes: 3074F (15.8k lines), 3008F (10.9k), 3078F (8.6k), 3079F (4.7k), 1159F (4.1k), about 47.5k lines.
2. **False-negative buckets.** When grouping the model's \~10.5k missed PA services (80% recall), quality codes were given their own rule (`^\d{4}F$`) and placed in bucket 1, "likely label noise".
3. **Root-cause check on the misses.** For the test misses, 1,930 of 1,939 quality-code misses carry an auth number shared with another service on the same visit; the other 9 are TRUE from a denial code on the claim.
4. **Claim-level auth confirmed.** On visits with 2+ services and an auth, 91.7% carry the same auth on every line and 0.2% carry different auths per service. The auth belongs to the claim, not the line.

## Evidence they are never PA

The definition and our own data point the same way: quality codes carry no payment and are PA only when another service on the claim is.

**Definition**

- AMA: Category II codes are "supplemental tracking codes that can be used for performance measurement" and "do not have a relative value associated with them" ([AMA](https://www.ama-assn.org/practice-management/criteria-cpt-category-ii-codes)).
- EmblemHealth, one of our payors: "No payment is associated with Category II CPT codes" and they "are used to report quality measures only" ([EmblemHealth](https://emblemhealth.com/providers/claims-corner/coding/category-ii-cpt-codes)).
- Category III codes (four digits + T) are emerging services that can need PA; the pattern `^\d{4}F$` deliberately excludes them.

**Data (std v4, 2025–2026)**

| Check | Result |
| --- | --- |
| How quality-code TRUE lines got TRUE | 20,000 via an auth shared with another service; 91 via a denial code; 7 via an auth not shared |
| Quality-code TRUE lines alone on a claim | 4 of \~6.4M quality lines |
| TRUE rate when labelled, per code | at most 0.9% for every one of the 84 codes |
| TRUE rate alone vs with other services (2026 sample) | 3008F 0.0% vs 0.1%; 1159F 0.0% vs 1.0%; 3075F 0.0% vs 0.8% |

## Size of the issue

Quality codes are almost absent before 2025 and about a third of all lines from 2025; relabelling them moves line- and service-level PA rates but leaves visit-level PA essentially unchanged.

**Raw files by year** (before any cleaning; 84 distinct codes across all years)

| Year | All lines | Quality lines | Share | Distinct codes | Quality TRUE | Quality AMBIGUOUS |
| --- | --- | --- | --- | --- | --- | --- |
| 2026 (to Jul) | 8,621,569 | 2,684,045 | 31.1% | 65 | 6,778 | 1,123,087 |
| 2025 | 10,099,438 | 3,738,097 | 37.0% | 71 | 13,320 | 1,377,953 |
| 2024 | 3,673,762 | 126,784 | 3.5% | 59 | 896 | 35,024 |
| 2023 | 2,638,365 | 29,238 | 1.1% | 45 | 31 | 10,973 |
| 2022 | 2,481,635 | 3,069 | 0.1% | 24 | 0 | 252 |
| **Total** | **27,514,769** | **6,581,233** | **23.9%** | **84** | **21,025** | **2,547,289** |

Top codes by lines: 3074F 1.39M, 3078F 1.31M, 3008F 1.19M, 1159F 0.66M, 3079F 0.56M, 3075F 0.34M, 1160F 0.27M, 3077F 0.22M, 3044F 0.17M, 1126F 0.13M.

**Training data as the pipeline prepares it** (2022–2026, AMBIGUOUS dropped, repeats removed, all filing orders)

|  | Keep (today) | Quality → FALSE | Quality excluded |
| --- | --- | --- | --- |
| Lines | 13,305,977 | 15,478,073 | 9,415,341 |
| PA lines | 536,610 | 515,956 | 515,956 |
| PA line rate | 4.03% | 3.33% | 5.48% |
| Services (CSN + CPT) | 12,613,003 | 14,329,432 | 8,844,522 |
| PA services | 494,776 | 475,513 | 475,513 |
| PA service rate | 3.92% | 3.32% | 5.38% |
| Visits | 5,434,035 | 5,735,094 | 4,781,193 |
| PA visits | 252,763 | 252,720 | 252,720 |
| PA visit rate | 4.65% | 4.41% | 5.29% |

Under "FALSE", the 2.17M quality lines that are AMBIGUOUS today stop being dropped, which is why lines rise. Any PA rate quoted from now on must name the mode.

**Test window (May–Jul 2026), services:** 1,949,656 in the benchmark, of which 448,521 (23%) are quality codes carrying 2,527 PA services (4.8% of all test PA).

## Risk review: did we miss genuine PA?

The chance of removing a genuine PA is low: at most about 100 lines and 43 visits are not explained by a shared auth, and none of them has been shown to be a real quality-code PA. Those small groups have not yet been inspected line by line.

**What we checked**

| Check | Result | Read |
| --- | --- | --- |
| Source of every quality-code TRUE line (2025–26) | 99.5% shared auth, 91 denial code, 7 own auth | Almost all inherited |
| Quality-code TRUE alone on a claim | 4 lines of \~6.4M | No standalone PA pattern |
| TRUE rate per code (all 84 codes) | max 0.9% | No code behaves like a PA service |
| PA visits whose only PA lines are quality codes | 43 of 252,763 (2022: 0, 2023: 0, 2024: 6, 2025: 23, 2026: 14) | The only place visit-level truth changes |
| Test misses that are quality codes | 1,930 of 1,939 shared auth, 9 denial code | Same pattern in the test window |

**Not yet checked (open)**

- The 7 own-auth and 91 denial-code quality-code TRUE lines: open the claims and see what else was billed and which codes were returned.
- The 43 visits that lose PA: likely visits where the real PA service was billed on another claim or another date. Not confirmed.
- Whether quality codes can appear in a prediction request. They are recorded during or after the visit, so probably not. If they can, answer them "not required" by rule before the model.

**Other risks**

- **Extract change in 2025.** Quality codes go from 3.5% of lines in 2024 to 37% in 2025, explaining 3.6M of the 6.4M jump in yearly lines. That points to a change in what the extract captures. Ask the data team.
- **Rule scope.** The rule is the code pattern only. A new Category II code is caught automatically, and Category III (####T) is never caught.
- **Reported numbers move.** Line and service PA rates, test-set size and FN counts change with the mode, so old and new figures are not directly comparable.

## What we changed: v6.0 dataset

v6.0 is v4 with four columns added; every row and original column is unchanged, and all counts match the analysis.

Location: `s3://cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v6/ai_prior_auth_09282026_v3_{year}_std.parquet`, with `v6_manifest.json` (counts per year). Built by `build_v6.py`.

| Column | Value |
| --- | --- |
| `IS_QUALITY_CODE` | 1 if `FINAL_CPT` matches `^\d{4}F$` |
| `LABEL_V6` | FALSE for quality codes, whatever the original label; otherwise a copy of `GROUP_LABEL` |
| `RELABEL_REASON` | `quality_code_cpt2` where `LABEL_V6` differs from `GROUP_LABEL` |
| `LABEL_VERSION` | `v6.0` |

**Validation**

- Row count of each written year equals v4 (checked by the build).
- TRUE before − TRUE after = TRUE→FALSE for every year: 547,317 → 526,292 TRUE lines (−21,025).
- AMBIGUOUS→FALSE = 2,547,289 lines, matching the analysis per year.
- Training on v6 with `PA_LABEL_COL=LABEL_V6` reproduced the analysis totals exactly (15,478,073 lines, 515,956 PA lines).

How each training mode reads v6: keep = `GROUP_LABEL`; FALSE = `LABEL_V6`; exclude = `GROUP_LABEL` where `IS_QUALITY_CODE = 0`.

## Open items

- [ ] Spot-check the \~98 quality-code TRUE lines not explained by a shared auth (7 own auth, 91 denial code)
- [ ] Look at the 43 visits whose only PA lines are quality codes
- [ ] Confirm with the integration owner whether quality codes can appear in a prediction request
- [ ] Ask the data team why line volumes and quality codes jump from 2025
- [ ] Store `v6_manifest.json` next to the v6 data

Spot check for the first two items (2026 shown; repeat for 2025):

```python
import pandas as pd, pyarrow.fs as pafs
p = "cli-ml-data-prod/datalake/ai-prior-auth/09282026/std_data_v6/ai_prior_auth_09282026_v3_2026_std.parquet"
d = pd.read_parquet(p, filesystem=pafs.S3FileSystem(), columns=["CSN", "FILING_ORDER_C", "FINAL_CPT", "GROUP_LABEL",
        "LINE_AUTHORIZATION_NUMBER", "CARC", "RARC", "IS_QUALITY_CODE", "PAYOR", "PLAN", "SERVICEDATE"])
d["T"] = d.GROUP_LABEL.astype(str).str.upper().eq("TRUE")
d["auth"] = d.LINE_AUTHORIZATION_NUMBER.astype(str).str.strip().replace({"nan": "", "None": ""})
q = d[(d.IS_QUALITY_CODE == 1) & d.T]
other = d[(d.IS_QUALITY_CODE == 0) & (d.auth != "")][["CSN", "FILING_ORDER_C", "auth"]].drop_duplicates()
q = q.merge(other.assign(shared=True), on=["CSN", "FILING_ORDER_C", "auth"], how="left")
odd = q[q.shared.isna()]                                    # quality TRUE not explained by a shared auth
print(len(odd)); print(odd[["CSN", "FINAL_CPT", "auth", "CARC", "RARC", "PAYOR", "PLAN"]].head(30))
v = d[d.T].groupby("CSN").IS_QUALITY_CODE.min()            # visits whose PA lines are all quality codes
print(d[d.CSN.isin(v[v == 1].index)].sort_values(["CSN", "SERVICEDATE"])[["CSN", "SERVICEDATE", "FINAL_CPT", "GROUP_LABEL", "auth", "CARC"]].head(60))
```

Model results for the three training modes are written up separately.
