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
from constants import TARGET_LABEL

_airport: str = "KMEM"

# load train and test data frame
train_df, val_df = mytools.get_train_and_test_ds(_airport, False)

dtrain = xgboost.DMatrix(train_df.drop(columns=[TARGET_LABEL]), label=train_df[TARGET_LABEL])
dtest = xgboost.DMatrix(val_df.drop(columns=[TARGET_LABEL]), label=val_df[TARGET_LABEL])

# obtain the path for the model
model_path: str = mytools.get_model_path(f"xgboost_regression_{_airport}.model")

_parameters = {
    "objective": "reg:squarederror",
    "eval_metric": "mae",
    "tree_method": "gpu_hist",
    "subsample": 0.1,
    "gamma": 1,
    "eta": 0.01,
}

evallist = [(dtrain, "train"), (dtest, "eval")]

num_round: int = 10000

model = xgboost.train(_parameters, dtrain, num_round, evallist, early_stopping_rounds=100)

print(f"Best mae for {_airport}: {model.best_score}\n")

# model.save_model(mytools.get_model_path(model_path))
