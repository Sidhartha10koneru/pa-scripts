bash run_feats.sh score > /dev/null
python -c "import pandas as pd; d=pd.read_csv('feats_confusion.csv'); print(d[(d['answer key']=='agreed')&(d['target recall']=='85%')].to_string(index=False))"
