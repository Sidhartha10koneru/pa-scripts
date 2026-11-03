o = L[L.segment == "OBX"].copy()
o["obx"] = o.column.str.extract(r"OBX#(\d+)")[0]
ident = o[o.column.str.contains(r"-3\.(1|2) /")].groupby(["file", "obx"]).value.agg(" | ".join)
vals = o[o.column.str.contains(r"-5(\.\d+)? /")].groupby(["file", "obx"]).value.agg(" | ".join)
ov = pd.concat([ident.rename("identifier"), vals.rename("value")], axis=1).reset_index()
print(ov[ov.identifier.str.contains(r"OR\.|ORDER|MRI|SURG", case=False, na=False)].to_string(index=False))

# everything in the SIU (scheduling) message
siu = L[L.msg.isin(L.loc[L.column.str.startswith("MSH-9.1") & (L.value == "SIU"), "msg"])]
print(siu.segment.value_counts())
