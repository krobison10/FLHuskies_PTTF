# M. De Cock; Mar 24, 2023
from datetime import timedelta
from pathlib import Path
import catboost as cb

from sklearn import tree
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMClassifier
from lightgbm import LGBMRegressor 
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder
from category_encoders import TargetEncoder
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIRECTORY = Path("./data")
airport = "KSEA"

train = pd.read_csv("KSEA_train_w_lamp.csv")
test = pd.read_csv("KSEA_test_w_lamp.csv")

train.columns = train.columns.to_series().apply(lambda x: x.strip())
test.columns = test.columns.to_series().apply(lambda x: x.strip())

# # make sure that the categorical features are encoded as strings
# cat_features = train.columns[np.where(train.dtypes != float)[0]].values.tolist()
# train[cat_features] = train[cat_features].astype(str)
# cat_features = test.columns[np.where(test.dtypes != float)[0]].values.tolist()
# test[cat_features] = test[cat_features].astype(str)

enc2 = OrdinalEncoder()
train["engine_enc"] = enc2.fit_transform(train[["aircraft_engine_class"]].values)
test["engine_enc"] = enc2.transform(test[["aircraft_engine_class"]].values)

enc3 = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value = -1)
train["aircraft_type_enc"] = enc3.fit_transform(train[["aircraft_type"]].values)
test["aircraft_type_enc"] = enc3.transform(test[["aircraft_type"]].values)

enc4 = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value = -1)
train["carrier_enc"] = enc4.fit_transform(train[["major_carrier"]].values)
test["carrier_enc"] = enc4.transform(test[["major_carrier"]].values)

enc5 = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value = -1)
train["flight_type_enc"] = enc5.fit_transform(train[["flight_type"]].values)
test["flight_type_enc"] = enc5.transform(test[["flight_type"]].values)

enc6 = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value = -1)
train["cloud_enc"] = enc6.fit_transform(train[["cloud"]].values)
test["cloud_enc"] = enc6.transform(test[["cloud"]].values)

features = ["aircraft_type_enc","flight_type_enc","carrier_enc","engine_enc","wind_gust", "cloud_ceiling", "cloud_enc", "lightning_prob", "precip", "wind_speed", "wind_direction", "temperature"]

X_train = train[features]

y_train = train["minutes_until_pushback"]

X_test = test[features]

y_test = test["minutes_until_pushback"]

# regressor = LGBMRegressor(objective="regression_l1")
# regressor.fit(X_train, y_train)

ensembleRegressor = cb.CatBoostRegressor(loss_function="MAE", task_type="GPU", n_estimators=8000)
ensembleRegressor.fit(X_train, y_train)

print("Finished training")

# y_pred = test["minutes_until_departure"] - 15
# print("Baseline:", mean_absolute_error(y_test, y_pred))

# y_pred = regressor.predict(X_train)
# print("Regression tree train error:", mean_absolute_error(y_train, y_pred))

y_pred = ensembleRegressor.predict(X_test)
print("Ensemble of tree regressors test error:", mean_absolute_error(y_test, y_pred))