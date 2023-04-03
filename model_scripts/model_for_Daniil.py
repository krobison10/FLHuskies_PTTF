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

offset = 4
cat_feature = [15,16,17,18,19,20,21,22,22]
cat_feature = [c - offset for c in cat_feature]
# print("Starting feature processing")
features = (train.columns.values.tolist())[offset:]
features.remove("departure_runway_actual")
# features = features + [s + enc for s in cat_feature]
# print(features)

X_train = train[features]

y_train = train["minutes_until_pushback"]

X_test = test[features]

y_test = test["minutes_until_pushback"]

# regressor = LGBMRegressor(objective="regression_l1")
# regressor.fit(X_train, y_train)

ensembleRegressor = cb.CatBoostRegressor(has_time=True, loss_function="MAE", task_type="GPU", n_estimators=1000)
ensembleRegressor.fit(X_train, y_train,cat_features=cat_feature,use_best_model=True)
print("Finished training")

# y_pred = test["minutes_until_departure"] - 15
# print("Baseline:", mean_absolute_error(y_test, y_pred))

# y_pred = regressor.predict(X_train)
# print("Regression tree train error:", mean_absolute_error(y_train, y_pred))

y_pred = ensembleRegressor.predict(X_test)
print("Ensemble of tree regressors test error:", mean_absolute_error(y_test, y_pred))