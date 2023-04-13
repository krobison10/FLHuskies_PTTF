#
# Kyler Robison
#
# Basic run of various models with the new validation technique.
#

import os
import pandas as pd
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import OrdinalEncoder

from sklearn.metrics import mean_absolute_error

# ---------------------------------------- LOAD ----------------------------------------

# specify a smaller list or just one airport in list for a single evaluation
airports = [
    "KATL",
    "KCLT",
    "KDEN",
    "KDFW",
    "KJFK",
    "KMEM",
    "KMIA",
    "KORD",
    "KPHX",
    "KSEA",
]

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

# whether to display train set MAE or not for each airport
TRAIN_MAE = False

# whether to use encoders for all airports, or encoders for each airport.
GLOBAL_ENCODERS = False # Absolutely leave this false, takes forever and uses a lot of memory

encoders = {}

# fits encoders over all airports training data
if GLOBAL_ENCODERS:
    df = pd.read_csv(os.path.join("..", "train_tables", "ALL_train.csv"), low_memory=False)
    for col in encoded_columns:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

        if col == "precip":
            df[col] = df[col].astype(str)

        encoders[col] = encoder.fit_transform(df[[col]])
    del df

predictions = pd.Series(dtype=object)
labels = pd.Series(dtype=object)


# ---------------------------------------- BASELINE ----------------------------------------

# # add columns representing standard and improved baselines to validation table
# val_df["baseline"] = val_df.apply(lambda row: max(row["minutes_until_etd"] - 15, 0), axis=1)
#
# # print performance of baseline estimates
# mae = mean_absolute_error(val_df["minutes_until_pushback"], val_df["baseline"])
# print(f"\nMAE with baseline: {mae:.4f}")


# ---------------------------------------- PROCESS ----------------------------------------

# df['minutes_until_etd'] = df['minutes_until_etd'].apply(lambda x: max(x, 0))

for airport in airports:
    print(f"Loading {airport} tables...")

    train_df = pd.read_csv(os.path.join("..", "train_tables", f"{airport}_train.csv"), low_memory=False)

    val_df = pd.read_csv(os.path.join("..", "validation_tables", f"{airport}_validation.csv"), low_memory=False)

    # need to make provisions for handling unknown values
    for col in encoded_columns:
        if col == "precip":
            train_df[col] = train_df[col].astype(str)
            val_df[col] = val_df[col].astype(str)

        if GLOBAL_ENCODERS:
            train_df[[col]] = encoders[col].transform(train_df[[col]])
            val_df[[col]] = encoders[col].transform(val_df[[col]])
        else:
            encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

            train_df[[col]] = encoder.fit_transform(train_df[[col]])

            val_df[[col]] = encoder.transform(val_df[[col]])

    X_train = train_df[input_features]
    y_train = train_df["minutes_until_pushback"]

    X_test = val_df[input_features]
    y_test = val_df["minutes_until_pushback"]

    print(f"Training on {airport}...")

    model = LGBMRegressor(objective="regression_l1")

    model.fit(X_train, y_train)

    if TRAIN_MAE:
        train_pred = model.predict(X_train)
        print(f"Train error: {mean_absolute_error(y_train, train_pred):.4f}")

    val_pred = model.predict(X_test)
    print(f"Validation error: {mean_absolute_error(y_test, val_pred):.4f}\n")

    predictions = pd.concat([predictions, pd.Series(val_pred.tolist())], ignore_index=True)
    labels = pd.concat([labels, pd.Series(y_test.tolist())], ignore_index=True)

if len(airports) > 1:
    print(f"MAE on all airports: {mean_absolute_error(predictions, labels):.4f}\n")
