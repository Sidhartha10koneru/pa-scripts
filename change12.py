nohup bash -c "SEED=7 bash run_feats.sh 6; TRAIN_YEARS='2025 2026' bash run_feats.sh 6" > run_next.log 2>&1 &


bash run_feats.sh score > /dev/null
python -c "import pandas as pd; d=pd.read_csv('feats_confusion.csv'); print(d[(d['answer key']=='agreed')&(d['target recall']=='85%')].to_string(index=False))"
