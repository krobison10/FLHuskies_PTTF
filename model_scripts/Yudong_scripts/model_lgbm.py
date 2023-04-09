#
# Author: Kyler Robison
# Modified by: Yudong Lin
# Basic run of various models with the new validation technique.
#

import lightgbm  # type: ignore
import matplotlib.pyplot as plt
import mytools
from joblib import dump  # type: ignore
from sklearn.metrics import mean_absolute_error  # type: ignore
from sklearn.preprocessing import OrdinalEncoder  # type: ignore

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
    "time": ["month", "day", "hour", "year", "weekday", "is_us_holiday", "minutes_until_etd"],
    "guif": ["airport", "gufi_flight_destination_airport", "gufi_flight_FAA_system"],
    "config": ["departure_runways"],
}

input_features: list[str] = []
for theFeatures in features.values():
    input_features += theFeatures

encoded_columns: list[str] = [
    "airport",
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

ignore_features: list[str] = []

for _ignore in ignore_features:
    input_features.remove(_ignore)
    if _ignore in encoded_columns:
        encoded_columns.remove(_ignore)

airports: tuple[str, ...] = ("KATL", "KCLT", "KDEN", "KDFW", "KJFK", "KMEM", "KMIA", "KORD", "KPHX", "KSEA", "ALL")

# load csv files

for airport in airports:
    train_df = mytools.get_train_tables(airport, remove_duplicate_gufi=False)
    val_df = mytools.get_validation_tables(airport, remove_duplicate_gufi=False)

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

    model: lightgbm.LGBMRegressor = lightgbm.LGBMRegressor(
        objective="regression_l1", device_type="gpu", num_leaves=8192, n_estimators=128
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(f"MAE on test data {airport}: {mean_absolute_error(y_test, y_pred):.4f}\n")

    lightgbm.plot_importance(model)
    plt.savefig(mytools.get_model_path(f"lgbm_{airport}_importance.png"), bbox_inches="tight")

    dump(model, mytools.get_model_path(f"lgbm_{airport}_model.joblib"))

    exit()
