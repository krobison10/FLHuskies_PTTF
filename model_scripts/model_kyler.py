#
# Kyler Robison
#
# Basic run of various models with the new validation technique.
#

import pandas as pd
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import OrdinalEncoder

from sklearn.metrics import mean_absolute_error

import math

# ---------------------------------------- LOAD ----------------------------------------

airport = "all"

train_df = pd.read_csv(f"../train_tables/{airport}_train.csv")
val_df = pd.read_csv(f"../validation_tables/{airport}_validation.csv")

# encode airports
encoder = OrdinalEncoder()
encoded_airports = encoder.fit_transform(train_df[["airport"]])
train_df["airport"] = encoded_airports

encoded_airports = encoder.transform(val_df[["airport"]])
val_df["airport"] = encoded_airports

input_features = ["minutes_until_etd", "airport"]


# ---------------------------------------- BASELINE ----------------------------------------

# add columns representing standard and improved baselines to validation table
val_df["baseline"] = val_df.apply(lambda row: max(row["minutes_until_etd"] - 15, 0), axis=1)

# print performance of baseline estimates
mae = mean_absolute_error(val_df["minutes_until_pushback"], val_df["baseline"])
print(f"\nMAE with baseline: {mae:.4f}")


# ---------------------------------------- PROCESS ----------------------------------------

# df['minutes_until_etd'] = df['minutes_until_etd'].apply(lambda x: max(x, 0))

X_train = train_df[input_features]
X_test = val_df[input_features]

y_train = train_df["minutes_until_pushback"]
y_test = val_df["minutes_until_pushback"]

# ---------------------------------------- TRAIN ----------------------------------------

print("\nTraining LightGBM Regressor")

model = LGBMRegressor(objective="regression_l1")

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"MAE on test data: {mean_absolute_error(y_test, y_pred):.4f}\n")


print("\nTraining CatBoost Regressor")

model = CatBoostRegressor(loss_function="MAE", n_estimators=500, silent=True, allow_writing_files=False)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"MAE on test data: {mean_absolute_error(y_test, y_pred):.4f}\n")
