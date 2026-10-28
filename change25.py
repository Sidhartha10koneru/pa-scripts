import lightgbm as lgb, pandas as pd
m = lgb.Booster(model_file="models/lgb_rules_full.txt")
X = apply_categories(claims, cats, feats)                 # the 4 claims, encoded as in training
contrib = pd.DataFrame(m.predict(X, pred_contrib=True), columns=feats + ["average"])
print(contrib.round(2))                                    # one row per claim, one column per feature

X_whatif = X.copy(); X_whatif.loc[3, "AGE"] = 41           # what-if: change only age on claim 4
print(m.predict(X.iloc[[3]]), "->", m.predict(X_whatif.iloc[[3]]))
