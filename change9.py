python -c "
import pandas as pd; pd.set_option('display.width',250); pd.set_option('display.max_colwidth',60)
f='plan_candidates.xlsx'; top=['UHC','AETNA','EMHECH','NYMVP','CIGNA']
p=pd.read_excel(f,'rules_plans_multi'); print(p[p.xref_id.isin(top)].to_string(index=False))
c=pd.read_excel(f,'multi_plan_candidates')
cols=['PAYOR','PLAN','claims','rules_payor','claim_segment','cand1_name','cand1_score','cand2_name','cand2_score','confidence']
for x in top: print('\n=====',x); print(c[c.rules_payor==x][cols].head(15).to_string(index=False))
"
