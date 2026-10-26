import optuna, lightgbm as lgb
from sklearn.metrics import average_precision_score
# train, update = TRAIN / UPDATE windows from pa_experiments (fit on train, score on update)
def objective(trial):
    params = dict(objective="binary", verbose=-1, seed=42,
        learning_rate=trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        num_leaves=trial.suggest_int("num_leaves", 31, 511, log=True),
        min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 20, 1000, log=True),
        feature_fraction=trial.suggest_float("feature_fraction", 0.5, 1.0),
        lambda_l2=trial.suggest_float("lambda_l2", 1e-3, 50, log=True),
        cat_smooth=trial.suggest_float("cat_smooth", 1, 100, log=True),
        cat_l2=trial.suggest_float("cat_l2", 1, 100, log=True),
        max_cat_threshold=trial.suggest_int("max_cat_threshold", 16, 256, log=True),
        min_data_per_group=trial.suggest_int("min_data_per_group", 20, 500, log=True))
    m = lgb.train(params, lgb.Dataset(Xtr, ytr), 2000, valid_sets=[lgb.Dataset(Xup, yup)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    p = m.predict(Xup)
    enc = pd.DataFrame({"y": yup.values, "p": p, "c": csn_up}).groupby("c").max()
    return average_precision_score(enc.y, enc.p)           # visit-level AP on Feb-Apr
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=40, timeout=3600)
print(study.best_value, study.best_params)
