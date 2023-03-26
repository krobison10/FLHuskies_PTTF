import os

import numpy as np
import pandas as pd  # type: ignore
from sklearn.ensemble import AdaBoostRegressor  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore
from sklearn.tree import DecisionTreeRegressor  # type: ignore
from sklearn.model_selection import cross_val_score  # type: ignore

DATA_DIR: str = os.path.dirname(__file__)

_data = pd.DataFrame = pd.read_csv(os.path.join(DATA_DIR, "..", "train_tables", "main.csv"), parse_dates=["timestamp"])

# fill nan with 0
_data = _data.fillna(0)

_data["month"] = _data.apply(lambda x: x.timestamp.month, axis=1)
_data["day"] = _data.apply(lambda x: x.timestamp.day, axis=1)
_data["hour"] = _data.apply(lambda x: x.timestamp.hour, axis=1)
_data["minute"] = _data.apply(lambda x: x.timestamp.minute, axis=1)
_data["weekday"] = _data.apply(lambda x: x.timestamp.weekday(), axis=1)

X = np.asarray(
    [
        _data["month"],
        _data["day"],
        _data["hour"],
        _data["minute"],
        _data["weekday"],
        _data["delay_3hr"],
        _data["delay_30hr"],
        _data["standtime_3hr"],
        _data["standtime_30hr"],
        _data["departure_runway"],
    ]
)

X = np.reshape(X, (X.shape[1], X.shape[0]))

y = np.asarray(_data["minutes_until_pushback"])

print("Make prediction using LinearRegression")
model_lr = LinearRegression()
result = cross_val_score(model_lr, X=X, y=y, scoring="neg_mean_absolute_error", cv=10)
print("average neg_mean_absolute_error for 10 cross_val_score:", np.average(result), ", best:", np.max(result))

print("Make prediction using DecisionTreeRegressor")
model_dtr = DecisionTreeRegressor()
result = cross_val_score(model_dtr, X=X, y=y, scoring="neg_mean_absolute_error", cv=10)
print("average neg_mean_absolute_error for 10 cross_val_score:", np.average(result), ", best:", np.max(result))

print("Make prediction using AdaBoostRegressor")
model_ada = AdaBoostRegressor()
result = cross_val_score(model_ada, X=X, y=y, scoring="neg_mean_absolute_error", cv=10)
print("average neg_mean_absolute_error for 10 cross_val_score:", np.average(result), ", best:", np.max(result))
