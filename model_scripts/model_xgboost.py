#
# Author: Yudong Lin
#
# A simple regression model implements with xgboost
#
import os
import numpy as np
import pandas as pd

import xgboost

_data_train = pd.DataFrame = pd.read_csv(
    os.path.join(os.path.dirname(__file__), "..", "train_tables", "KSEA_train.csv"),
    parse_dates=["timestamp"],
    dtype={"minutes_until_etd": int, "minutes_until_pushback": int},
)

_data_test = pd.DataFrame = pd.read_csv(
    os.path.join(os.path.dirname(__file__), "..", "validation_tables", "KSEA_validation.csv"),
    parse_dates=["timestamp"],
    dtype={"minutes_until_etd": int, "minutes_until_pushback": int},
)

features: tuple[str, ...] = ("minutes_until_etd",)

X_train = np.asarray([_data_train[_col] for _col in features])
X_test = np.asarray([_data_test[_col] for _col in features])

X_train = np.reshape(X_train, (X_train.shape[1], X_train.shape[0]))
X_test = np.reshape(X_test, (X_test.shape[1], X_test.shape[0]))

y_train = np.asarray(_data_train["minutes_until_pushback"])
y_test = np.asarray(_data_test["minutes_until_pushback"])

dtrain = xgboost.DMatrix(X_train, label=y_train)
dtest = xgboost.DMatrix(X_test, label=y_test)

_parameters = {"objective": "reg:squarederror", "eval_metric": "mae", "tree_method": "gpu_hist", "gamma": 1, "eta": 0.01, "subsample": 0.2}

evallist = [(dtrain, "train"), (dtest, "eval")]

num_round = 10000
result = xgboost.train(_parameters, dtrain, num_round, evallist, early_stopping_rounds=50)
