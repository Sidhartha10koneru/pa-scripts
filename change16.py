v9 update: option C passenger labels

What we built

The rule: never relabel a CPT at a payer if that payer has at least one PA line for that CPT billed alone. Protected pairs are found on training data only (before May 2026).
5,719 CPT + payer pairs are protected.
Passengers: 23,531 in v8 → 11,861 in v9. 11,670 lines are back to PA.
PA lines: 526,292 (no passenger fix) → 502,761 (v8) → 514,431 (v9).
The build only undoes v8 flips; nothing new was flipped. Count checks passed for every year.

Results (seed 42; seed 7 in brackets where we ran it)

Answer key	No passenger fix	v8 passenger	v9 option C
Original labels	0.653 / 52,356	0.652 / 55,920	0.650 / 55,706
v8 passenger labels	0.666 / 41,415	0.675 / 36,676	0.669 / 39,891
Option C labels	0.658 / 48,180	0.662 / 45,605 (0.663 / 44,326)	0.661 / 47,271 (0.660 / 45,227)
Services where all labels agree	0.668 / 40,880	0.675 / 36,633	0.670 / 39,644 (s7: 38,621)

What it shows

The v8 model beats the option C model even when scored on option C’s own labels: about 1,700 fewer false alarms (seed 42) and about 900 fewer (seed 7).
On services where every label definition agrees (the fairest comparison), v8 has 36,633 false alarms vs 39,644 for option C. That is about 3,000 (8%) more for option C.
Real PA caught above the cut-off: v8 86.0%, option C 85.1%, no fix 84.1%.
The 11,670 kept lines:
The option C model ranks only 21.9% of them above the cut-off, even though it was trained on them as PA. The model with no fix ranks 20.0% above it, and v8 2.1%.
So training on them as PA catches very few of them, and it adds noise that costs precision on everything else.
Inference: with our 5 features these lines look the same as the non-PA lines of the same CPT, so the model can’t separate them.
