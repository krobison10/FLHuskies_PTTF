# Daniil Filienko
#
# Running the model emsemble with Kyler's Train Split for Trevor
# to attain accuracy values for individual airports and overall
from Yudong_scripts.mytools import *
import matplotlib.pyplot as plt
from train_test_split import *
import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns
from lightgbm import LGBMRegressor, Dataset
import lightgbm as lgb
import pickle
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_absolute_error
import argparse
from pathlib import Path

# ---------------------------------------- MAIN ----------------------------------------
DATA_DIRECTORY = Path("full_tables")
OUTPUT_DIRECTORY = Path("./models/Daniil_models")

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


for airport in airports:
    # replace this path with the locations of the full tables for each airport if necessary
    df = pd.read_csv(DATA_DIRECTORY / f"{airport}_full.csv", parse_dates=["gufi_flight_date", "timestamp"])
    # df.rename(columns = {'wind_direction':'wind_direction_cat', 'cloud_ceiling':'cloud_ceiling_cat', 'visibility':'visibility_cat'}, inplace = True)

    for c in df.columns:
        col_type = df[c].dtype
        if col_type == "object" or col_type == "string" or "cat" in c:
            df[c] = df[c].astype("category")

    offset = 2
    features_all = [
        "minutes_until_etd",
        "deps_3hr",
        "arrs_3hr",
        "deps_taxiing",
        "arrs_taxiing",
        "exp_deps_15min",
        "exp_deps_30min",
        "delay_3hr",
        "standtime_3hr",
        "dep_taxi_3hr",
        "arr_taxi_3hr",
        "1h_ETDP",
        "departure_runways",
        "arrival_runways",
        "temperature",
        "wind_direction",
        "wind_speed",
        "wind_gust",
        "cloud_ceiling",
        "cloud",
        "lightning_prob",
        "gufi_flight_major_carrier",
        "gufi_flight_destination_airport",
        "gufi_timestamp_until_etd",
        "year",
        "month",
        "day",
        "hour",
        "minute",
    ]
    features_remove = ("gufi_flight_date", "minutes_until_pushback")
    features = [
        x
        for x in features_all
        if x not in features_remove
        and not ("precip" in x or "lamp" in x or "engine" in x or "faa" in x or "ratio" in x)
    ]
    cat_features = get_clean_categorical_columns()

    # evaluating individual airport accuracy
    print(f"Training LIGHTGBM model for {airport}\n")
    X_train = df[features]
    y_train = df["minutes_until_pushback"]
    train_data = lgb.Dataset(X_train, label=y_train)

    fit_params = {
        "objective": "regression_l1",  # Type of task (regression)
        "metric": "mae",  # Evaluation metric (mean squared error)
        "num_leaves": 1024 * 8,
        "n_estimators": 128,
    }

    regressor = LGBMRegressor(**fit_params)

    regressor.fit(X_train, y_train)

    print("Finished training")
    # # SAVING THE MODEL
    filename = f"model_{airport}_submission.sav"
    pickle.dump(regressor, open(OUTPUT_DIRECTORY / filename, "wb"))
    print("Saved the model for the airport: ", airport)

print(features)
exit()
