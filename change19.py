import pandas as pd, pyarrow.fs as pafs
import pa_base_adapter as pb, pa_experiments as pe
from sklearn.metrics import average_precision_score
from pycaret.classification import setup, add_metric, compare_models

cols = ["CSN","GROUP_LABEL","SERVICEDATE","FILING_ORDER_C","FINAL_CPT","FINAL_MODIFIERS","PLACE_OF_SERVICE_CODE","PAYOR","PLAN"]
d = pe.add_dates(pb.load([2025, 2026], cols, 0.1, pafs.S3FileSystem()))
train, update, test = pe.split(d, "2026-02-01", "2026-05-01", "2026-08-01")

feats = ["FINAL_CPT","FINAL_MODIFIERS","PLACE_OF_SERVICE_CODE","PAYOR_CLEAN","PAYOR_PLAN_CLEAN"]
tr = train[feats + ["y"]].astype({c: str for c in feats}).sample(200_000, random_state=42)
up = update[feats + ["y"]].astype({c: str for c in feats}).sample(80_000, random_state=42)

setup(data=tr, test_data=up, target="y", categorical_features=feats,
      data_split_shuffle=False, fold=3, fold_shuffle=False, session_id=42, verbose=False)
add_metric("ap", "AP", average_precision_score, target="pred_proba")
best = compare_models(include=["lr","dt","rf","et","lightgbm","xgboost","catboost"], sort="AP", n_select=3)
