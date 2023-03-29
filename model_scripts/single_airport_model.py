#
# Kyler Robison
#
# Basic run of various models with the new validation technique.
#

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer

import math

# ---------------------------------------- LOAD ----------------------------------------

airport = "KSEA"

train_df = pd.read_csv(f"../train_tables/{airport}_train.csv")
val_df = pd.read_csv(f"../validation_tables/{airport}_val.csv")

# input_features = ['minutes_until_etd', 'departure_runway']
input_features = ["minutes_until_etd"]


# ---------------------------------------- BASELINE ----------------------------------------

# add columns representing standard and improved baselines to validation table
val_df["baseline"] = val_df.apply(lambda row: max(row["minutes_until_etd"] - 15, 0), axis=1)
val_df["improved_baseline"] = val_df.apply(lambda row: round(max(row["minutes_until_etd"] - 13, 0)), axis=1)

# print performance of baseline estimates
mae = mean_absolute_error(val_df["minutes_until_pushback"], val_df["baseline"])
print(f"\nMAE with baseline: {mae:.4f}")
mae = mean_absolute_error(val_df["minutes_until_pushback"], val_df["improved_baseline"])
print(f"MAE with improved baseline: {mae:.4f}\n")


# ---------------------------------------- PROCESS ----------------------------------------

# df['minutes_until_etd'] = df['minutes_until_etd'].apply(lambda x: max(x, 0))

X_train = train_df[input_features]
X_test = val_df[input_features]

y_train = train_df["minutes_until_pushback"]
y_test = val_df["minutes_until_pushback"]

# Impute missing values with mean
imputer = SimpleImputer(strategy="mean")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)


# ---------------------------------------- TRAIN ----------------------------------------

print("Training linear regression model")
model = LinearRegression()
model.fit(X_train, y_train)

print("Intercept:", model.intercept_)
print("Slope:", model.coef_)

y_pred = model.predict(X_test)
print(f"MAE on test data: {mean_absolute_error(y_test, y_pred):.4f}")


print("\nTraining decision tree regressor")
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"MAE on test data: {mean_absolute_error(y_test, y_pred):.4f}\n")
