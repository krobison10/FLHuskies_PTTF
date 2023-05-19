#
# Author: Yudong Lin
#
# A simple regression model implements with xgboost
#

import mytools
import xgboost
from constants import ALL_AIRPORTS, TARGET_LABEL

for _airport in ALL_AIRPORTS:
    # load train and test data frame
    train_df, val_df = mytools.get_train_and_test_ds(_airport)

    train_ds: xgboost.DMatrix = xgboost.DMatrix(
        train_df.drop(columns=[TARGET_LABEL]), label=train_df[TARGET_LABEL], enable_categorical=True
    )
    test_ds: xgboost.DMatrix = xgboost.DMatrix(
        val_df.drop(columns=[TARGET_LABEL]), label=val_df[TARGET_LABEL], enable_categorical=True
    )

    _parameters: dict = {
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "tree_method": "gpu_hist",
        "subsample": 0.1,
        "gamma": 1,
        "eta": 0.01,
    }

    num_round: int = 10000

    model = xgboost.train(
        _parameters, train_ds, num_round, evals=[(train_ds, "train"), (test_ds, "eval")], early_stopping_rounds=100
    )

    print(f"Best mae for {_airport}: {model.best_score}\n")

    model.save_model(mytools.get_model_path(f"xgboost_regression_{_airport}.json"))
