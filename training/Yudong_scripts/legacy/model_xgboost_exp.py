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

overwrite: bool = False

_airport: str = "KSEA"

_data_train: pd.DataFrame = mytools.get_train_tables(_airport)
_data_test: pd.DataFrame = mytools.get_validation_tables(_airport)

features: dict[str, tuple[str, ...]] = {
    "lamp": (
        "temperature",
        "wind_direction",
        "wind_speed",
        "wind_gust",
        "cloud_ceiling",
        "visibility",
        "cloud",
        "lightning_prob",
        "precip",
    ),
    "mfs": ("aircraft_engine_class", "aircraft_type", "major_carrier", "flight_type"),
    "time": ("standtime_3hr", "standtime_30hr", "month", "day", "hour", "weekday"),
}

mytools.encodeStrFeatures(_data_train, _data_test, *features["mfs"])
mytools.encodeStrFeatures(_data_train, _data_test, "cloud", "lightning_prob", "precip")

for _category in features:
    # load training and validation data
    X_train: np.ndarray = np.asarray([_data_train[_col] for _col in features[_category]], dtype="float32")
    X_test: np.ndarray = np.asarray([_data_test[_col] for _col in features[_category]], dtype="float32")

    X_train = np.reshape(X_train, (X_train.shape[1], X_train.shape[0]))
    X_test = np.reshape(X_test, (X_test.shape[1], X_test.shape[0]))

    y_train: np.ndarray = np.asarray(_data_train["minutes_until_pushback"])
    y_test: np.ndarray = np.asarray(_data_test["minutes_until_pushback"])

    dtrain = xgboost.DMatrix(X_train, label=y_train)
    dtest = xgboost.DMatrix(X_test, label=y_test)

    # obtain the path for the model
    model_path: str = mytools.get_model_path(f"xgboost_regression_{_airport}_{_category}.model")
    # don overwrite existing model unless specified
    if not os.path.exists(model_path) or overwrite is True:
        _parameters = {
            "objective": "reg:squarederror",
            "eval_metric": "mae",
            "tree_method": "gpu_hist",
            "gamma": 1,
            "eta": 0.01,
            "subsample": 0.2,
        }

        evallist = [(dtrain, "train"), (dtest, "eval")]

        num_round: int = 10000

        model = xgboost.train(_parameters, dtrain, num_round, evallist, early_stopping_rounds=10)

        print(f"Best mae for {_category}: {model.best_score}\n")

        model.save_model(model_path)
    else:
        model = xgboost.Booster()
        model.load_model(model_path)

    _data_train[f"xgboost_{_category}"] = model.predict(dtrain)
    _data_train[f"xgboost_{_category}"] = _data_train[f"xgboost_{_category}"].astype(int)
    _data_test[f"xgboost_{_category}"] = model.predict(dtest)
    _data_test[f"xgboost_{_category}"] = _data_train[f"xgboost_{_category}"].astype(int)

    for _col in features[_category]:
        _data_train = _data_train.drop(_col, axis=1)
        _data_test = _data_test.drop(_col, axis=1)

_data_train.to_csv(mytools.get_train_tables_path().replace(".csv", "_xgboost.csv"), index=False)
_data_train.to_csv(mytools.get_validation_tables_path().replace(".csv", "_xgboost.csv"), index=False)
