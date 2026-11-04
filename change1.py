nohup python -u fn_root_cause.py --buckets 1 --cover 0.9 --out rc_b1 > run_rc_b1.log 2>&1 &


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
    ["2/3 code denial", "4 therapy", "1 claim-level auth", "5 own auth"], "no TRUE line found")
b["family"] = np.select([b.FINAL_CPT.str.fullmatch(r"\d{4}F"), b.FINAL_CPT.str.fullmatch(r"99[2-4]\d\d|G2211"),
                         b.FINAL_CPT.str.fullmatch(r"8\d{4}|36415")], ["quality F code", "E/M", "lab"], "other")
print(len(b)); print(b.problem.value_counts())
print(pd.crosstab(b.family, b.problem, margins=True))
print(b[b.problem == "1 claim-level auth"].groupby("FINAL_CPT").size().sort_values(ascending=False).head(15))


