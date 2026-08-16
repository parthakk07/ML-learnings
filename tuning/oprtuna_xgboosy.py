# %%
import pandas as pd
import numpy as np
import optuna
import xgboost as xgb
from optuna_integration import XGBoostPruningCallback
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
# %%
X, y = load_iris(return_X_y=True)
X
# %%
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.1 , random_state=42)
X_train
# %%
from sklearn.metrics import accuracy_score
def objective(trial):
    param = {
            'verbosity': 0,
            'objective': 'multi:softprob',
            'num_class': 3,
            'eval_metric': 'mlogloss',  # Ensure that the eval_metric is specified here
            'booster': 'gbtree',
            'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
            'eta': trial.suggest_float('eta', 0.01, 0.3),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.4, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'n_estimators': 300,
        }
    dtrain=xgb.DMatrix(X_train,label=Y_train)
    dtest=xgb.DMatrix(X_test,label=Y_test)
    pruning_callback = XGBoostPruningCallback(trial, "eval-mlogloss")  # Match the metric name in the evals list

    bst = xgb.train(
            param,
            dtrain,
            num_boost_round=300,
            evals=[(dtrain, "train"), (dtest, "eval")],  # Ensure the eval datasets and names are specified
            early_stopping_rounds=30,
            callbacks=[pruning_callback]
        )
    preds=bst.predict(dtest)
    best_preds = [int(np.argmax(line)) for line in preds]
    accuracy=accuracy_score(Y_test,best_preds)
    return accuracy

study = optuna.create_study(direction='maximize', pruner=optuna.pruners.SuccessiveHalvingPruner())
study.optimize(objective, n_trials=50)


# %%
print(f"Best trial: {study.best_trial.params}")
print(f"Best accuracy: {study.best_value}")
# %%
