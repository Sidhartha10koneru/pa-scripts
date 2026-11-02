print([c for c in x.columns if any(k in c.upper() for k in ["AUTH", "LEVEL", "CARC", "RARC", "OVERRIDE", "SOURCE", "N_"])])

auth = x["LINE_AUTHORIZATION_NUMBER"].astype(str).str.strip().str.upper()      # adjust name
no_auth_text = auth.str.fullmatch(r"NAR|N/?A|NONE|NO ?AUTH.*|NOT ?REQ.*|NO ?PA.*", na=False)
blank_codes = x.CARC.astype(str).str.strip().isin(["", "nan", "None"]) & x.RARC.astype(str).str.strip().isin(["", "nan", "None"])
x["false_source"] = np.where(x.y == 1, "TRUE",
                    np.where(no_auth_text, "FALSE: explicit no-auth",
                    np.where(blank_codes, "FALSE: blank codes", "FALSE: other")))
x["pat_has_pa"] = x.groupby("PAT_ID").y.transform("max") == 1
print(pd.crosstab(x.false_source, [x.pat_has_pa, x.prior_pa], margins=True))
