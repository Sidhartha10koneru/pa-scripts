import pandas as pd, re
f = pd.read_csv("rc_mixed/fn_causes.csv", dtype=str)
for c in ["T2_passenger","T3_open_auth","T4_subgroup","T5_change","T6_too_little"]:
    f[c] = f[c].eq("True")
print(pd.read_csv("rc_mixed/cause_rollup.csv").to_string())          # primary cause (the table that scrolled off)

# shared auth, typed by what the auth is shared with (all FN with a shared auth, not only T2)
drug  = lambda c: bool(re.fullmatch(r"[JQ]\d{4}", c))
admin = lambda c: bool(re.fullmatch(r"963[67]\d|964[01]\d|96372|20610|2055\d", c))
img   = lambda c: bool(re.fullmatch(r"7\d{4}|930(15|16|17|18)|933(0[3-8]|5[01])|7849\d|7845\d", c))
proc  = lambda c: bool(re.fullmatch(r"[1-6]\d{4}", c))
def share_type(r):
    w = [x for x in str(r.t2_shared_with).split(",") if x and x != "nan"]
    if not w: return "not shared"
    me = r.FINAL_CPT
    if (drug(me) and any(admin(x) or drug(x) or proc(x) for x in w)) or (admin(me) and any(drug(x) for x in w)):
        return "drug + admin/procedure"
    if any(img(x) or proc(x) for x in w): return "next to imaging/procedure"
    if any(drug(x) for x in w): return "next to a drug"
    return "visit-level auth (labs / E&M only)"
f["share_type"] = f.apply(share_type, axis=1)
print(pd.crosstab(f.share_type, f.T3_open_auth, margins=True))
print(f[f.share_type=="visit-level auth (labs / E&M only)"].FINAL_CPT.value_counts().head(10))
print(pd.crosstab(f.cause, f.share_type, margins=True))
