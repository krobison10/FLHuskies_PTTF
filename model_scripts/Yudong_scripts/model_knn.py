#
# Author: Yudong Lin
#
# A model that predict push back time using knn model
#

import mytools
import numpy as np
import pandas as pd  # type: ignore
from joblib import dump
from sklearn.metrics import mean_absolute_error  # type: ignore
from sklearn.neighbors import KNeighborsRegressor  # type: ignore

_data_train: pd.DataFrame = mytools.get_train_tables().drop_duplicates(subset=["gufi"])
_data_test: pd.DataFrame = mytools.get_validation_tables().drop_duplicates(subset=["gufi"])

_data_train = mytools.applyAdditionalTimeBasedFeatures(_data_train)
_data_test = mytools.applyAdditionalTimeBasedFeatures(_data_test)

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
    X_train: np.ndarray = np.asarray([_data_train[_col] for _col in features[_category]], dtype="float32")
    X_test: np.ndarray = np.asarray([_data_test[_col] for _col in features[_category]], dtype="float32")

    X_train = np.reshape(X_train, (X_train.shape[1], X_train.shape[0]))
    X_test = np.reshape(X_test, (X_test.shape[1], X_test.shape[0]))

    y_train: np.ndarray = np.asarray(_data_train["minutes_until_pushback"])
    y_test: np.ndarray = np.asarray(_data_test["minutes_until_pushback"])

    model_knn = KNeighborsRegressor(50)

    model_knn.fit(X_train, y_train)

    result = model_knn.predict(X_test)

    print(f"Validation mae for {_category}: {mean_absolute_error(y_test, result)}\n")

    dump(model_knn, mytools.get_model_path(f"knn_{_category}_model.joblib"))
