import re
grp = r.CARC.astype(str).str.upper().str.strip().str.replace(r"^(CO|PR|OA|PI|CR)[-\s]*", "", regex=True)
r["carc_code"] = grp.where(grp.str.fullmatch(r"\d+"))                         # numeric CARC only; MA15/N381 etc. excluded
r["denial"] = r.carc_code.isin(["15","39","197","198","210","284","296","302"]) | \
              r.RARC.astype(str).str.upper().str.contains(r"(?<![A-Z])(M62|N54)\b", regex=True, na=False)
print("raw CARC values counted as PA denial:\n", r.loc[r.carc_code.isin(["15","39","197","198","210","284","296","302"]), "CARC"].astype(str).value_counts().head(15).to_string())
print("MA15-type values now excluded:", r.CARC.astype(str).str.contains("MA15").sum())

# then rerun the rest of the audit cell from "ln = r.groupby(...)" unchanged


own_rate = ln.assign(own=ln.evidence.isin(["own denial","own unique auth"])).groupby("cpt").own.mean()
s = ln[ln.auth.notna() & ln.auth_shared].copy()
s["cpt_own_rate"] = s.cpt.map(own_rate)
s["anchor"] = s.groupby(["CSN","auth"]).cpt_own_rate.transform("max") == s.cpt_own_rate
pas = s[~s.anchor].cpt.value_counts().head(20)
print("Passenger CPTs most often sharing someone else's auth:\n", pas.to_string())
