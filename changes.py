rp = rules_df[["PAYOR_CODE","PAYOR_NAME"]].dropna().astype(str).drop_duplicates()
up = rp.PAYOR_NAME.str.upper()
def suggest(t):
    hit = rp[up.str.contains(rf"\b{re.escape(t)}", regex=True)]
    return "; ".join((hit.PAYOR_CODE + " | " + hit.PAYOR_NAME).head(5))
tok["rules_payor_candidates"] = tok["token"].map(suggest)
with pd.ExcelWriter("token_review.xlsx") as w:
    tok.to_excel(w, sheet_name="tokens", index=False)
    rp.to_excel(w, sheet_name="rules_payors", index=False)
