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

# ---------------------------------------- LOAD ----------------------------------------

# airport = "ALL"
# airport = "legacy_ALL"
airport = "ALL"

train_df = pd.read_csv(f"../train_tables/{airport}_train.csv")
# train_df = pd.read_csv(f"../full_tables/master_full.csv")

val_df = pd.read_csv(f"../validation_tables/{airport}_validation.csv")

input_features = [
    "airport",
    "minutes_until_etd",
    "temperature",
    "wind_direction",
    "wind_speed",
    "wind_gust",
    "cloud_ceiling",
    "visibility",
    "cloud",
    "lightning_prob",
    "precip",
    "aircraft_engine_class",
    "aircraft_type",
    "major_carrier",
    "flight_type",
]

encoded_columns = [
    "airport",
    "cloud",
    "lightning_prob",
    "precip",
    "aircraft_engine_class",
    "aircraft_type",
    "major_carrier",
    "flight_type",
]

encoders = {}

# need to make provisions for handling unknown values
for col in encoded_columns:
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    if col == "precip":
        train_df[col] = train_df[col].astype(str)
        val_df[col] = val_df[col].astype(str)

    encoded_col = encoder.fit_transform(train_df[[col]])
    train_df[[col]] = encoded_col

    encoded_col = encoder.transform(val_df[[col]])
    val_df[[col]] = encoded_col

# ---------------------------------------- BASELINE ----------------------------------------

# # add columns representing standard and improved baselines to validation table
# val_df["baseline"] = val_df.apply(lambda row: max(row["minutes_until_etd"] - 15, 0), axis=1)
#
# # print performance of baseline estimates
# mae = mean_absolute_error(val_df["minutes_until_pushback"], val_df["baseline"])
# print(f"\nMAE with baseline: {mae:.4f}")


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

exit()

print("\nTraining CatBoost Regressor")

model = CatBoostRegressor(loss_function="MAE", n_estimators=500, silent=True, allow_writing_files=False)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"MAE on test data: {mean_absolute_error(y_test, y_pred):.4f}\n")
