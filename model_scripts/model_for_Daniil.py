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

DATA_DIRECTORY = Path("./train_tables")
airport = "KSEA"

train = pd.read_csv(DATA_DIRECTORY / "main_KSEA_prescreened.csv")
train = train.sort_values(by=['gufi'])
test = train.iloc[round(train.shape[0]*0.9):]
train = train.iloc[:round(train.shape[0]*0.9)]

# # make sure that the categorical features are encoded as strings
cat_feature = train.columns[np.where(train.dtypes != float)[0]].values.tolist()
train[cat_feature] = train[cat_feature].astype(str)
test[cat_feature] = test[cat_feature].astype(str)

# enc2 = OrdinalEncoder()
# train["engine_enc"] = enc2.fit_transform(train[["aircraft_engine_class"]].values)
# test["engine_enc"] = enc2.transform(test[["aircraft_engine_class"]].values)

# enc3 = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value = -1)
# train["aircraft_type_enc"] = enc3.fit_transform(train[["aircraft_type"]].values)
# test["aircraft_type_enc"] = enc3.transform(test[["aircraft_type"]].values)

# enc4 = OrdinalEncoder()
# train["carrier_enc"] = enc4.fit_transform(train[["major_carrier"]].values)
# test["carrier_enc"] = enc4.transform(test[["major_carrier"]].values)

# enc5 = OrdinalEncoder()
# train["flight_type_enc"] = enc5.fit_transform(train[["flight_type"]].values)
# test["flight_type_enc"] = enc5.transform(test[["flight_type"]].values)

# enc6 = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value = -1)
# train["cloud_enc"] = enc6.fit_transform(train[["cloud"]].values)
# test["cloud_enc"] = enc6.transform(test[["cloud"]].values)

cat_feature = ["cloud","lightning_prob","aircraft_engine_class","aircraft_type"
                ,"major_carrier","flight_type","gufi_end_label","precip","wind_direction"]
enc = "_enc"
print("Starting feature processing")
features = (train.columns.values.tolist())[4:14]
features = features + [s + enc for s in cat_feature]
print(features)

enc0 = OrdinalEncoder()
train["lightning_prob_enc"] = enc0.fit_transform(train[["lightning_prob"]].values)
test["lightning_prob_enc"] = enc0.transform(test[["lightning_prob"]].values)

enc1 = OrdinalEncoder()
train["cloud_enc"] = enc1.fit_transform(train[["cloud"]].values)
test["cloud_enc"] = enc1.transform(test[["cloud"]].values)

enc2 = OrdinalEncoder()
train["aircraft_engine_class_enc"] = enc2.fit_transform(train[["aircraft_engine_class"]].values)
test["aircraft_engine_class_enc"] = enc2.transform(test[["aircraft_engine_class"]].values)

enc3 = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value = -1)
train["aircraft_type_enc"] = enc3.fit_transform(train[["aircraft_type"]].values)
test["aircraft_type_enc"] = enc3.transform(test[["aircraft_type"]].values)

enc4 = OrdinalEncoder()
train["major_carrier_enc"] = enc4.fit_transform(train[["major_carrier"]].values)
test["major_carrier_enc"] = enc4.transform(test[["major_carrier"]].values)

enc5 = OrdinalEncoder()
train["flight_type_enc"] = enc5.fit_transform(train[["flight_type"]].values)
test["flight_type_enc"] = enc5.transform(test[["flight_type"]].values)

enc6 = OrdinalEncoder()
train["gufi_end_label_enc"] = enc6.fit_transform(train[["gufi_end_label"]].values)
test["gufi_end_label_enc"] = enc6.transform(test[["gufi_end_label"]].values)

enc7 = OrdinalEncoder()
train["wind_direction_enc"] = enc7.fit_transform(train[["wind_direction"]].values)
test["wind_direction_enc"] = enc7.transform(test[["wind_direction"]].values)

enc8 = OrdinalEncoder()
train["precip_enc"] = enc8.fit_transform(train[["precip"]].values)
test["precip_enc"] = enc8.transform(test[["precip"]].values)

X_train = train[features]

y_train = train["minutes_until_pushback"]

X_test = test[features]

y_test = test["minutes_until_pushback"]

# regressor = LGBMRegressor(objective="regression_l1")
# regressor.fit(X_train, y_train)

ensembleRegressor = cb.CatBoostRegressor(has_time=True, loss_function="MAE", task_type="GPU", n_estimators=21000)
ensembleRegressor.fit(X_train, y_train,use_best_model=True)
print("Finished training")

# y_pred = test["minutes_until_departure"] - 15
# print("Baseline:", mean_absolute_error(y_test, y_pred))

# y_pred = regressor.predict(X_train)
# print("Regression tree train error:", mean_absolute_error(y_train, y_pred))

y_pred = ensembleRegressor.predict(X_test)
print("Ensemble of tree regressors test error:", mean_absolute_error(y_test, y_pred))