import numpy as np, pandas as pd
THERAPY = r"^(97\d{3}|9894[0-2]|92507|92526)$"
parts = []
for name in ["rc_b1", "rc_b2", "rc_b3", "rc_mixed"]:
    t = pd.read_csv(f"{name}/fn_causes.csv", dtype=str).assign(run=name)
    parts.append(t)
A = pd.concat(parts, ignore_index=True)
A["shared"] = A.t2_shared_with.fillna("").ne("")
A["therapy"] = A.FINAL_CPT.str.match(THERAPY)

# 1) therapy: is it only NYCE PPO, or other plans too?
th = A[A.therapy]
print(th.groupby("PAYOR_PLAN_CLEAN").size().sort_values(ascending=False).head(15))

# 2) where PA rose after training: which plans, and when did PA start?
cells = pd.concat([pd.read_csv(f"{n}/cells.csv", dtype=str).assign(run=n) for n in ["rc_b2", "rc_b3", "rc_mixed"]])
ch = cells[cells.T5_change.eq("True")].assign(fn=lambda t: t.fn_services.astype(int))
print(ch.groupby("PAYOR_PLAN_CLEAN").fn.sum().sort_values(ascending=False).head(15))
print(ch.groupby("first_pa_month").fn.sum().sort_index().tail(18))
