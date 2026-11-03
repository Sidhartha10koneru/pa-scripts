a = s[s.AUTH != ""].copy()
a["auth_kind"] = "number/code"
a.loc[a.AUTH.str.contains(r"REFERRAL|\bREF\b", regex=True), "auth_kind"] = "referral text"
a.loc[a.AUTH.str.contains(r"NO\s+PRIOR|NOT\s+REQ|NPN|NPAR|UNIVERSAL", regex=True), "auth_kind"] = "no-auth / placeholder text"
a.loc[a.AUTH.str.fullmatch(r"(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])20\d\d"), "auth_kind"] = "date"
a.loc[a.AUTH.str.fullmatch(r"[A-Z ]+") & a.auth_kind.eq("number/code"), "auth_kind"] = "other text"
print(pd.crosstab(a.auth_kind, a.L, margins=True))
print(a[a.auth_kind != "number/code"].AUTH.value_counts().head(25))
