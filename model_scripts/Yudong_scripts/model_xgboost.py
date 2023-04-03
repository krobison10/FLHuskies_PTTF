#
# Author: Yudong Lin
#
# A simple regression model implements with xgboost
#

import os
import mytools
import numpy as np
import pandas as pd  # type: ignore
import xgboost

_data_train: pd.DataFrame = mytools.get_train_tables().drop_duplicates(subset=["gufi"])
_data_test: pd.DataFrame = mytools.get_validation_tables().drop_duplicates(subset=["gufi"])

features: tuple[str, ...] = ("wind_direction", "wind_gust", "temperature", "delay_3hr", "delay_30hr", "standtime_3hr", "standtime_30hr")

X_train = np.asarray([_data_train[_col] for _col in features])
X_test = np.asarray([_data_test[_col] for _col in features])

X_train = np.reshape(X_train, (X_train.shape[1], X_train.shape[0]))
X_test = np.reshape(X_test, (X_test.shape[1], X_test.shape[0]))

y_train = np.asarray(_data_train["minutes_until_pushback"])
y_test = np.asarray(_data_test["minutes_until_pushback"])

dtrain = xgboost.DMatrix(X_train, label=y_train)
dtest = xgboost.DMatrix(X_test, label=y_test)

_parameters = {
    "objective": "reg:squarederror",
    "eval_metric": "mae",
    "tree_method": "gpu_hist",
    "gamma": 1,
    "eta": 0.01,
    "subsample": 0.2,
    "booster": "dart",
}

evallist = [(dtrain, "train"), (dtest, "eval")]

num_round = 10000

model = result = xgboost.train(_parameters, dtrain, num_round, evallist, early_stopping_rounds=10)

model.save_model(os.path.join(os.path.dirname(__file__), "..", "..", "models", "xgboost_regression.model"))
