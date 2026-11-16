A line becomes a passenger (TRUE → FALSE) only if all five hold:

It’s TRUE in the original label because of an auth number, not a “no auth” value.
The same auth number is on another CPT of the same claim (claim = CSN + filing order).
Its CPT is rarely PA on its own: when it’s the only CPT on a claim, its PA rate is under 2%, based on at least 100 such lines in training data only.
The CPT–payer pair is not a payer exception, i.e. not a payer where that CPT is often PA on its own (e.g. referral or HMO plans).
The claim keeps another real PA line, the “owner” of the auth. With no owner left, the line stays TRUE and is flagged VISIT_LEVEL_AUTH, because it could be a referral.


cat v8_draft_manifest.json
for f in P_size_by_threshold.csv R2_payor_exceptions.csv; do echo "== $f"; head -25 v6_impact_owner/$f; done
wc -l v6_impact_owner/P_list_thr2.csv
grep -A12 "OWNER RULE\|R1 under owner" run_v6_impact*.log | head -40
cat v8d_compare/pax.csv
