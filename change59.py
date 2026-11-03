import pandas as pd
c = pd.read_csv("sep_mixed_v3/cells.csv")
print(pd.crosstab(c.category, c.A_label_flag, values=c.fn_services, aggfunc="sum", margins=True))
print(c[c.category.str.startswith("B")].sort_values("fn_services", ascending=False)
       [["FINAL_CPT","PAYOR_PLAN_CLEAN","fn_services","AUC_oot","A_label_flag","inflation_x","top_features"]].to_string())
