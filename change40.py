# 1) N54x false matches
rr = r.RARC.astype(str).str.upper()
print("contains N54/M62:", rr.str.contains("M62|N54").sum(),
      "| exact token:", rr.str.contains(r"(?<![A-Z0-9])(?:M62|N54)(?![0-9])", regex=True).sum())
print(rr[rr.str.contains("N54\d", regex=True)].value_counts().head(10).to_string())

# 2) how much becomes AMBIGUOUS, and which codes cause it
cs, rs = r.CARC.astype(str).str.strip(), rr.str.strip()
blank = lambda s: s.isin(["", "NAN", "NONE", "nan", "None"])
amb = ~(cs.isin(["15","39","197","198","210","284","296","302"]) | rs.str.contains("M62|N54")) & ~(blank(cs) & blank(rs))
print("raw rows AMBIGUOUS at level 2:", amb.mean().round(3))
print(r.loc[amb, "CARC"].astype(str).value_counts().head(15).to_string())

# 3) in the final std data: AMBIGUOUS share per CPT for cells WITH PA vs without
