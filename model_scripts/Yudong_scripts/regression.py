#
# Author: Yudong Lin
#
# This script is used to experiment with some traditional regression models
# and find what features is/are better
#

import os

import mytools
import numpy as np
import pandas as pd  # type: ignore
from sklearn.ensemble import AdaBoostRegressor  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore
from sklearn.model_selection import cross_val_score  # type: ignore
from sklearn.neighbors import KNeighborsRegressor  # type: ignore
from sklearn.tree import DecisionTreeRegressor  # type: ignore

DATA_DIR: str = os.path.dirname(__file__)

_data = pd.DataFrame = mytools.get_train_tables()

# fill nan with 0
_data = _data.fillna(0)

features: tuple[str, ...] = ("minutes_until_etd",)
X = np.asarray([_data[_col] for _col in features], dtype="float32")

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

print("Make prediction using Knn")
model_knn = KNeighborsRegressor()
result = cross_val_score(model_knn, X=X, y=y, scoring="neg_mean_absolute_error", cv=10)
print("average neg_mean_absolute_error for 10 cross_val_score:", np.average(result), ", best:", np.max(result))
