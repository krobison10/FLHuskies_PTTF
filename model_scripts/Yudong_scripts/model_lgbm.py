#
# Author: Kyler Robison
# Modified by: Yudong Lin
# Basic run of various models with the new validation technique.
#

import argparse
import json
import os
from datetime import datetime

import lightgbm  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import mytools
import pandas as pd  # type: ignore
from joblib import dump  # type: ignore
from sklearn.metrics import mean_absolute_error  # type: ignore
from sklearn.preprocessing import OrdinalEncoder  # type: ignore

hyperparameter: dict = {
    "num_leaves": 1024 * 4,
    "n_estimators": 128,
    "boosting_type": "gbdt",
    "ignore_features": ["gufi_flight_FAA_system", "is_us_holiday"],
}

features: dict[str, list[str]] = {
    "lamp": [
        "temperature",
        "wind_direction",
        "wind_speed",
        "wind_gust",
        "cloud_ceiling",
        "visibility",
        "cloud",
        "lightning_prob",
        "precip",
    ],
    "mfs": ["aircraft_engine_class", "aircraft_type", "major_carrier", "flight_type"],
    "time": [
        "month",
        "day",
        "hour",
        "minute",
        "year",
        "weekday",
        "is_us_holiday",
        "minutes_until_etd",
        "gufi_timestamp_until_etd",
    ],
    "guif": ["gufi_flight_destination_airport", "gufi_flight_FAA_system"],
    "config": ["departure_runways"],
}

input_features: list[str] = []
for theFeatures in features.values():
    input_features += theFeatures

encoded_columns: list[str] = [
    "cloud",
    "lightning_prob",
    "precip",
    "aircraft_engine_class",
    "aircraft_type",
    "major_carrier",
    "flight_type",
    "departure_runways",
    "gufi_flight_destination_airport",
    "gufi_flight_FAA_system",
    "is_us_holiday",
]

for _ignore in hyperparameter["ignore_features"]:
    input_features.remove(_ignore)
    if _ignore in encoded_columns:
        encoded_columns.remove(_ignore)

# using argparse to parse the argument from command line
parser: argparse.ArgumentParser = argparse.ArgumentParser()
parser.add_argument("-a", help="airport")
args: argparse.Namespace = parser.parse_args()


ALL_AIRPORTS: tuple[str, ...] = ("KATL", "KCLT", "KDEN", "KDFW", "KJFK", "KMEM", "KMIA", "KORD", "KPHX", "KSEA", "ALL")

airports: tuple[str, ...] = ALL_AIRPORTS if args.a is None else (str(args.a).upper(),)

model_records_path: str = mytools.get_model_path(f"model_records.json")
model_records: dict[str, dict] = {}
if os.path.exists(model_records_path):
    with open(model_records_path, "r", encoding="utf-8") as f:
        model_records = dict(json.load(f))

TARGET_LABEL: str = "minutes_until_pushback"

for airport in airports:
    # check if the same hyperparameter has been used before
    same_setup_mae: float = -1
    input_features.sort()
    if airport not in model_records:
        model_records[airport] = {}
    else:
        for value in model_records[airport].values():
            if (
                value["num_leaves"] == hyperparameter["num_leaves"]
                and value["n_estimators"] == hyperparameter["n_estimators"]
                and value["boosting_type"] == hyperparameter["boosting_type"]
                and sorted(value["features"]) == input_features
            ):
                same_setup_mae = value["mae"]
                break
        if same_setup_mae > 0:
            print(f"Same setup found for airport {airport} found with mae {same_setup_mae}, skip!")
            continue

    train_df: pd.DataFrame = mytools.get_train_tables(airport, remove_duplicate_gufi=False)
    train_df.drop(columns=[col for col in train_df if col not in input_features and col != TARGET_LABEL], inplace=True)
    val_df: pd.DataFrame = mytools.get_validation_tables(airport, remove_duplicate_gufi=False)
    val_df.drop(columns=[col for col in train_df if col not in input_features and col != TARGET_LABEL], inplace=True)

    # need to make provisions for handling unknown values
    for col in encoded_columns:
        encoder: OrdinalEncoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

        encoded_col = encoder.fit_transform(train_df[[col]])
        train_df[[col]] = encoded_col

        encoded_col = encoder.transform(val_df[[col]])
        val_df[[col]] = encoded_col

    # df['minutes_until_etd'] = df['minutes_until_etd'].apply(lambda x: max(x, 0))

    X_train = train_df[input_features]
    X_test = val_df[input_features]

    y_train = train_df[TARGET_LABEL]
    y_test = val_df[TARGET_LABEL]

    # train model
    model: lightgbm.LGBMRegressor = lightgbm.LGBMRegressor(
        hyperparameter["boosting_type"],
        hyperparameter["num_leaves"],
        n_estimators=hyperparameter["n_estimators"],
        objective="regression_l1",
        device_type="gpu",
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_train)
    mae: float = round(mean_absolute_error(y_train, y_pred), 4)
    print(f"MAE on train data {airport}: {mae}")
    y_pred = model.predict(X_test)
    mae = round(mean_absolute_error(y_test, y_pred), 4)
    print(f"MAE on validation data {airport}: {mae}")

    # record model information
    model_name: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_records[airport][model_name] = {"mae": mae, "features": input_features}
    model_records[airport][model_name].update(hyperparameter)

    # plot the graph that shows importance
    lightgbm.plot_importance(model, ignore_zero=False)
    plt.savefig(mytools.get_model_path(f"lgbm_{airport}_{model_name}_importance.png"), bbox_inches="tight")

    # if the model is the best, the save it
    if (
        "best" not in model_records[airport]
        or model_records[airport]["best"]["mae"] > model_records[airport][model_name]["mae"]
    ):
        print("The best result so far, saved!")
        model_records[airport]["best"] = model_records[airport][model_name]
        model_records[airport]["best"]["achieve_at"] = model_name
        dump(model, mytools.get_model_path(f"lgbm_{airport}_model.joblib"))

    with open(model_records_path, "w", encoding="utf-8") as f:
        json.dump(model_records, f, indent=4, ensure_ascii=False, sort_keys=True)

    if airport == "ALL":
        for theAirport in ALL_AIRPORTS[: len(ALL_AIRPORTS) - 1]:
            val_airport_df: pd.DataFrame = val_df.loc[val_df.airport == theAirport]
            X_test = val_airport_df[input_features]
            y_test = val_airport_df[TARGET_LABEL]
            y_pred = model.predict(X_test)
            mae = round(mean_absolute_error(y_test, y_pred), 4)
            print(f"--------------------------------------------------")
            print(f"MAE when apply cumulative model on validation data for airport {airport}: {mae}")
            individual_model_best_mae = model_records[airport]["best"]["mae"]
            print(f"Compare to individual model's best current best {individual_model_best_mae},")
            if individual_model_best_mae > mae:
                print("Cumulative model is better.")
            elif individual_model_best_mae == mae:
                print("They are the same.")
            else:
                print("Individual model is better.")
