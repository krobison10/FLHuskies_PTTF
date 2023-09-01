#
# Daniil Filienko
#
# Running the model emsemble with Kyler's Train Split for Trevor
# to attain accuracy values for individual airports and overall

import matplotlib.pyplot as plt
import pandas as pd
import pickle
import torch
import numpy as np
from pathlib import Path
import seaborn as sns
from lightgbm import LGBMRegressor, Dataset
from sklearn.preprocessing import OrdinalEncoder

from sklearn.metrics import mean_absolute_error

# ---------------------------------------- MAIN ----------------------------------------
DATA_DIRECTORY = Path("full_tables")


def load_encoder(assets_directory):
    """Load all model assets from disk."""
    encoder = None
    with open(assets_directory + "/encoders.pickle", "rb") as fp:
        encoder = pickle.load(fp)

    return encoder


def load_model(assets_directory):
    model = None
    with open(assets_directory + "/model.pkl", "rb") as fp:
        model = pickle.load(fp)
    return model


def encode_df(_df: pd.DataFrame, encoded_columns: list, int_columns: list, encoders) -> pd.DataFrame:
    for column in encoded_columns:
        try:
            _df[column] = encoders[column].transform(_df[[column]])
        except Exception as e:
            print(e)
            print(column)
            print(_df.shape)
    for column in int_columns:
        try:
            _df[column] = _df[column].astype("int")
        except Exception as e:
            print(e)
            print(column)
    return _df


int_columns = [
    "deps_3hr",
    "deps_30hr",
    "deps_taxiing",
    "exp_deps_15min",
    "exp_deps_30min",
    "minute",
    "month",
    "day",
    "hour",
    "year",
    "weekday",
    "minutes_until_etd",
    "temperature",
    "wind_direction",
    "wind_speed",
    "wind_gust",
    "cloud_ceiling",
    "visibility",
    "gufi_timestamp_until_etd",
]

encoded_columns = [
    "cloud",
    "lightning_prob",
    "precip",
    # "gufi_flight_major_carrier",
    "gufi_flight_destination_airport",
    "aircraft_engine_class",
    "aircraft_type",
    "major_carrier",
    "flight_type",
    "airport",
    "airline",
    "departure_runways",
    "arrival_runways",
]

features = [
    # "gufi",
    # "gufi_flight_major_carrier",
    "airline",
    "airport",
    "deps_3hr",
    "deps_30hr",
    # "arrs_3hr",
    # "arrs_30hr",
    "deps_taxiing",
    # "arrs_taxiing",
    "exp_deps_15min",
    "exp_deps_30min",
    "standtime_30hr",
    "dep_taxi_30hr",
    "1h_ETDP",
    # "arr_taxi_30hr",
    "minute",
    "gufi_flight_destination_airport",
    "month",
    "day",
    "hour",
    "year",
    "weekday",
    "minutes_until_etd",
    "aircraft_engine_class",
    "aircraft_type",
    "major_carrier",
    "flight_type",
    "temperature",
    "wind_direction",
    "wind_speed",
    "wind_gust",
    "cloud_ceiling",
    "visibility",
    "cloud",
    "lightning_prob",
    "precip",
    "gufi_timestamp_until_etd",
    "departure_runways",
    "arrival_runways",
]

AIRLINES = [
    "AAL",
    "AJT",
    "ASA",
    "ASH",
    "AWI",
    "DAL",
    "EDV",
    "EJA",
    "ENY",
    "FDX",
    "FFT",
    "GJS",
    "GTI",
    "JBU",
    "JIA",
    "NKS",
    "PDT",
    "QXE",
    "RPA",
    "SKW",
    "SWA",
    "SWQ",
    "TPA",
    "UAL",
    "UPS",
]

AIRPORTS = [
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


def plotImp(model, X, airport, num=20, fig_size=(40, 20)):
    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": X.columns})
    plt.figure(figsize=fig_size)
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title("LightGBM Features (avg over folds)")
    plt.tight_layout()
    plt.savefig(f"lgbm_importances_{airport}_daniil.png")


all_train = []
all_val = []
for airline in AIRLINES:
    airline_df = []
    airline_valdf = []
    for airport in AIRPORTS:
        try:
            dfs = pd.read_csv(
                f"train_tables/{airport}/{airline}_train.csv",
                parse_dates=["timestamp"],
                dtype={"precip": str},
            )

            dfs_val = pd.read_csv(
                f"validation_tables/{airport}/{airline}_validation.csv",
                parse_dates=["timestamp"],
                dtype={"precip": str},
            )
        except FileNotFoundError:
            continue
        airline_df.append(dfs)
        airline_valdf.append(dfs_val)
    if len(airline_df) == 0:
        print(f"{airline}")
        continue

    train_df = pd.concat(airline_df)
    val_df = pd.concat(airline_valdf)
    if train_df.shape[0] == 0:
        continue
    all_train.append(train_df)
    all_val.append(val_df)


def train():
    train_df = pd.concat(all_train)
    val_df = pd.concat(all_val)
    encoder = load_encoder("assets")
    train_df = encode_df(train_df, encoded_columns, int_columns, encoder)
    val_df = encode_df(val_df, encoded_columns, int_columns, encoder)
    # FOR FINAL TRAINING
    # train_df.append(val_df)

    # ---------------------------------------- BASELINE ----------------------------------------
    # add columns representing standard and improved baselines to validation table
    val_df["baseline"] = val_df.apply(lambda row: max(row["minutes_until_etd"] - 15, 0), axis=1)

    # print performance of baseline estimates
    mae = mean_absolute_error(val_df["minutes_until_pushback"], val_df["baseline"])
    print(f"\nMAE with baseline: {mae:.4f}")

    # evaluating individual airport accuracy is currently disabled
    print(f"Training LIGHTGBM model\n")
    X_train = train_df[features]
    X_test = val_df[features]
    y_train = train_df["minutes_until_pushback"]
    y_test = val_df["minutes_until_pushback"]

    fit_params = {
        "eval_metric": "MAE",
        "verbose": 100,
        "feature_name": "auto",  # that's actually the default
        "categorical_feature": "auto",  # that's actually the default
    }

    ensembleRegressor = LGBMRegressor(objective="regression_l1")

    ensembleRegressor.fit(X_train, y_train, **fit_params)
    # ensembleRegressor.fit(X_train, y_train,cat_features=cat_features,use_best_model=True)

    # ensembleRegressor.fit(X_train, y_train, **fit_params)
    y_pred = ensembleRegressor.predict(X_test)
    with open("assets/model.pkl", "wb") as f:
        pickle.dump(ensembleRegressor, f)

    print("Finished training")
    print(f"MAE on total test data: {mean_absolute_error(y_test, y_pred):.4f}\n")

    # plotImp(ensembleRegressor, X_test, airport=airport)


train()
exit()
