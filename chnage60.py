o = L[L.segment == "OBX"].copy()
o["obx"] = o.column.str.extract(r"OBX#(\d+)")[0]
ident = o[o.column.str.contains(r"-3\.1 /")].set_index(["file", "msg", "obx"]).value.rename("id")
val = o[o.column.str.contains(r"-5(\.1)? /")].groupby(["file", "msg", "obx"]).value.agg(",".join)
when = o[o.column.str.contains(r"-14 /")].set_index(["file", "msg", "obx"]).value.rename("obs_time")
cpt = pd.concat([ident, val.rename("value"), when], axis=1).reset_index()
cpt = cpt[cpt.id.str.contains("CPT", case=False, na=False)]
msg = L[L.column.str.startswith(("MSH-9.2", "MSH-7 ", "PV1-3", "IN1#0-4"))].pivot_table(index=["file", "msg"], columns="column", values="value", aggfunc="first")
print(cpt.merge(msg, left_on=["file", "msg"], right_index=True).to_string(index=False))
