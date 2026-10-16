P = out.PAYOR.fillna("").str.upper().str.strip()
name = (out.PAYOR.fillna("") + " " + out.PLAN.fillna("")).str.upper()
seg = pd.Series(None, index=out.index, dtype=object)
seg[P.isin(["MEDICARE", "RAILROAD MEDICARE"]) | name.str.contains(r"^MEDICARE .*RAILROAD")] = "Original Medicare"
seg[seg.isna() & name.str.contains(r"SUPPLEMENT|MEDIGAP")] = "Medicare supplement"
seg[seg.isna() & name.str.contains(r"\bESSENTIAL\b")] = "Essential Plan"
seg[seg.isna() & name.str.contains(r"WORKER|ONE CALL|NO.?FAULT|\bTPL\b")] = "Workers comp / auto / TPL"
seg[seg.isna() & (out.LINE_OF_BUSINESS_share >= 0.9)] = out.lob_segment
kw = out.keyword_segment.where(~((out.keyword_segment == "Original Medicare") & ~P.isin(["MEDICARE", "RAILROAD MEDICARE"])), "Medicare Advantage")
out["proposed_segment"] = seg.fillna(kw).fillna(out.lob_segment).fillna("Review")
