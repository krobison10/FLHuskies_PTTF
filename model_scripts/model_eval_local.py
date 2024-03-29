# Daniil Filienko
#
# Running the model emsemble with Kyler's Train Split for Trevor
# to attain accuracy values for individual airports and overall

"""Alternative implementation of a traditional local eval"""
# @author:Daniil Filienko
from pathlib import Path
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import lightgbm as lgb
import pickle
import argparse
from sklearn.preprocessing import OrdinalEncoder
from train_test_split import *

DATA_DIRECTORY = Path("./full_tables")
DATA_DIRECTORY_TRAIN = Path("./train_tables")
DATA_DIRECTORY_VAL = Path("./validation_tables")
OUTPUT_DIRECTORY = Path("./models/Daniil_models")

parser: argparse.ArgumentParser = argparse.ArgumentParser()
parser.add_argument("-s", help="save the model")
args: argparse.Namespace = parser.parse_args()


def plotImp(model, X, airport="ALL", airline="ALL", num=20, fig_size=(40, 20)):
    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": X.columns})
    plt.figure(figsize=fig_size)
    sns.set(font_scale=3)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title("LightGBM Features (avg over folds)")
    plt.tight_layout()
    plt.savefig(f"lgbm_importances_{airline}at_{airport}_global.png")


AIRPORTS = ["KATL", "KCLT", "KDEN", "KDFW", "KJFK", "KMEM", "KMIA", "KORD", "KPHX", "KSEA"]

print("Started")
# train = pd.read_csv(DATA_DIRECTORY_TRAIN / f"ALL_train.csv", parse_dates=["gufi_flight_date","timestamp"])
# val = pd.read_csv(DATA_DIRECTORY_VAL / f"ALL_validation.csv", parse_dates=["gufi_flight_date","timestamp"])

features_remove = ("gufi_flight_date", "minutes_until_pushback", "timestamp", "gufi")

# # Preventing GUFI from being an attribute to analyze
# offset = 2
# features_all = (train.columns.values.tolist())[offset:(len(train.columns.values))]
# features_all_val = (val.columns.values.tolist())[offset:(len(val.columns.values))]

# features_remove = ("gufi_flight_date","minutes_until_pushback")
# features = [x for x in features_all if x not in features_remove]
# features_val = ["minutes_until_pushback","airport"]

feats: list[str] = [
    "minutes_until_etd",
    "deps_3hr",
    "deps_30hr",
    "arrs_3hr",
    "arrs_30hr",
    "deps_taxiing",
    "arrs_taxiing",
    "exp_deps_15min",
    "exp_deps_30min",
    "delay_30hr",
    "standtime_30hr",
    "dep_taxi_30hr",
    "arr_taxi_30hr",
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
    "weekday",
    "feat_5_gufi",
    "feat_5_estdep_next_30min",
    "feat_5_estdep_next_60min",
    "feat_5_estdep_next_180min",
    "feat_5_estdep_next_1400min",
    "aircraft_type",
    "major_carrier",
    "visibility",
    "flight_type",
]

cat_feats = [
    "departure_runways",
    "arrival_runways",
    "cloud",
    "lightning_prob",
    # "precip",
    # "gufi_flight_number",
    "gufi_flight_major_carrier",
    "gufi_flight_destination_airport",
    # "gufi_flight_FAA_system",
    # "aircraft_engine_class",
    "aircraft_type",
    "major_carrier",
    "flight_type",
    # "isdeparture",
    "feat_5_gufi",
]

int_features = [
    "feat_5_estdep_next_30min",
    "feat_5_estdep_next_60min",
    "feat_5_estdep_next_180min",
    "feat_5_estdep_next_1400min",
]

features_val = ["minutes_until_pushback"]

y_tests = [0]
y_preds = [0]
for airport in AIRPORTS:
    train, val = split(table=pd.read_csv(DATA_DIRECTORY / f"{airport}_full.csv", parse_dates=["timestamp"]), save=False)

    print("Generated a shared dataframe")

    encoders = dict()

    for c in int_features:
        train[c] = pd.to_numeric(train[c], errors="coerce")
        val[c] = pd.to_numeric(val[c], errors="coerce")

    for c in cat_feats:
        train[c] = train[c].astype(str)
        val[c] = val[c].astype(str)
        encoders[c] = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        train[c] = encoders[c].fit_transform(np.array(train[c]).reshape(-1, 1))
        train[c] = train[c].astype("int")
        val[c] = encoders[c].transform(np.array(val[c]).reshape(-1, 1))
        val[c] = val[c].astype("int")

    X_train = train[feats]
    y_train = train[features_val]

    X_val = val[feats]
    y_val = val[features_val]

    train_data = lgb.Dataset(X_train, label=y_train)

    fit_params = {
        "objective": "regression_l1",  # Type of task (regression)
        "metric": "mae",  # Evaluation metric (mean squared error)
        "num_leaves": 1024 * 4,
        "n_estimators": 128,
        "learning_rate": 0.05,
    }

    regressor = LGBMRegressor(**fit_params)

    regressor = lgb.train(fit_params, train_data)

    y_pred = regressor.predict(X_val)
    y_tests = np.concatenate((y_tests, y_val))
    y_preds = np.concatenate((y_preds, y_pred))

    # # SAVING THE MODEL
    save_table_as: str = "save" if args.s is None else str(args.s)
    if save_table_as == "save":
        filename = f"model_{airport}.sav"
        pickle.dump(regressor, open(OUTPUT_DIRECTORY / filename, "wb"))

    print(f"Regression tree train error for {airport}:", mean_absolute_error(y_val, y_pred))
    # plotImp(regressor, X_val)

print(f"MAE on all test data: {mean_absolute_error(y_tests, y_preds):.4f}\n")

exit()

"""Old implementation of a traditional local eval"""
for airport in airports:
    # replace this path with the locations of the full tables for each airport if necessary
    df = pd.read_csv(DATA_DIRECTORY / f"{airport}_full.csv", parse_dates=["gufi_flight_date", "timestamp"])
    # df.rename(columns = {'wind_direction':'wind_direction_cat', 'cloud_ceiling':'cloud_ceiling_cat', 'visibility':'visibility_cat'}, inplace = True)
    val_df = pd.DataFrame()

    train_df = df
    df["precip"] = df["precip"].astype(str)

    # Doing Ordinal Encoding for specified features
    ENCODER: dict[str, OrdinalEncoder] = get_encoder(airport, train_df, val_df)
    for col in ENCODED_STR_COLUMNS:
        if col != "year":
            train_df[[col]] = ENCODER[col].transform(train_df[[col]])

    cat_features = get_clean_categorical_columns()
    for c in train_df.columns:
        if any(c in x for x in cat_features):
            train_df[c] = train_df[c].astype("category")

    features = [
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

    # evaluating individual airport accuracy
    print(f"Training LIGHTGBM model for {airport}\n")
    X_train = train_df[features]
    y_train = train_df["minutes_until_pushback"]
    train_data = lgb.Dataset(X_train, label=y_train)

    fit_params = {
        "objective": "regression_l1",  # Type of task (regression)
        "metric": "mae",  # Evaluation metric (mean squared error)
        "num_leaves": 1024 * 8,
        "n_estimators": 128,
    }

    regressor = LGBMRegressor(**fit_params)

    regressor.fit(X_train, y_train)

    # # SAVING THE MODEL
    save_table_as: str = "save" if args.s is None else str(args.s)
    if save_table_as == "save":
        filename = f"model_{airport}.sav"
        pickle.dump(regressor, open(OUTPUT_DIRECTORY / filename, "wb"))
        print("Saved the model for the airport: ", airport)


print(features)
# y_tests = np.hstack(y_tests)
# y_pred = np.hstack(y_preds)
print(f"MAE on all test data: {mean_absolute_error(y_tests, y_preds):.4f}\n")


exit()
