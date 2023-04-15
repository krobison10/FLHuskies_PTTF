import json
import os
from datetime import datetime
from typing import Any

import lightgbm as lgb  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import mytools
import optuna
import pandas as pd  # type: ignore
from joblib import dump  # type: ignore
from sklearn.metrics import mean_absolute_error  # type: ignore
from sklearn.preprocessing import OrdinalEncoder  # type: ignore


def _train(
    trial, _airport, X_train, X_test, y_train, y_test, _encoded_columns, _model_records_ref, model_records_save_to
):
    dtrain = lgb.Dataset(X_train, label=y_train)

    params: dict[str, Any] = {
        "boosting_type": "gbdt",
        "objective": "regression_l1",
        "device_type": "gpu",
        "verbosity": -1,
        "num_leaves": trial.suggest_int("num_leaves", 1024, 1024 * 8),
        "n_estimators": trial.suggest_int("n_estimators", 64, 64 * 3),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }

    model = lgb.train(params, dtrain, categorical_feature=_encoded_columns)

    y_pred = model.predict(X_train)
    train_mae: float = round(mean_absolute_error(y_train, y_pred), 4)
    print(f"MAE on train data {_airport}: {train_mae}")
    y_pred = model.predict(X_test)
    test_mae = round(mean_absolute_error(y_test, y_pred), 4)
    print(f"MAE on validation data {_airport}: {test_mae}")

    # record model information
    model_name: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    _model_records_ref[_airport][model_name] = {
        "train_mae": train_mae,
        "val_mae": test_mae,
        "features": X_train.columns.values.tolist(),
    }
    _model_records_ref[_airport][model_name].update(params)

    # plot the graph that shows importance
    lgb.plot_importance(model, ignore_zero=False)
    plt.savefig(mytools.get_model_path(f"lgbm_{_airport}_{model_name}_importance.png"), bbox_inches="tight")

    # if the model is the best, the save it
    if (
        "best" not in _model_records_ref[_airport]
        or _model_records_ref[_airport]["best"]["val_mae"] > _model_records_ref[_airport][model_name]["val_mae"]
    ):
        if "best" in _model_records_ref[_airport]:
            print(
                f'The best val result so far (previous best: {_model_records_ref[_airport]["best"]["val_mae"]}), saved!'
            )
        _model_records_ref[_airport]["best"] = _model_records_ref[_airport][model_name]
        _model_records_ref[_airport]["best"]["achieve_at"] = model_name
        dump(model, mytools.get_model_path(f"lgbm_{_airport}_model.joblib"))
    else:
        print(f'Worse than previous best: {_model_records_ref[_airport]["best"]["val_mae"]})')

    with open(model_records_save_to, "w", encoding="utf-8") as f:
        json.dump(_model_records_ref, f, indent=4, ensure_ascii=False, sort_keys=True)

    return train_mae + test_mae


if __name__ == "__main__":
    TARGET_LABEL: str = "minutes_until_pushback"

    ignore_features = [
        "gufi",
        "timestamp",
        "airport",
        "gufi_flight_date",
        "isdeparture",
        "dep_ratio",
        "arr_ratio",
        "gufi_flight_number",
        "precip",
    ]

    encoded_columns: list[str] = [
        "cloud",
        "lightning_prob",
        "precip",
        "aircraft_engine_class",
        "aircraft_type",
        "major_carrier",
        "flight_type",
        "departure_runways",
        "arrival_runways",
        "gufi_flight_destination_airport",
        "gufi_flight_FAA_system",
        "gufi_flight_major_carrier",
    ]

    for _ignore in ignore_features:
        if _ignore in encoded_columns:
            encoded_columns.remove(_ignore)

    ALL_AIRPORTS: tuple[str, ...] = (
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
        "ALL",
    )

    for airport in ALL_AIRPORTS:
        train_df: pd.DataFrame = mytools.get_train_tables(airport, remove_duplicate_gufi=False)
        train_df.drop(columns=ignore_features, inplace=True)

        val_df: pd.DataFrame = mytools.get_validation_tables(airport, remove_duplicate_gufi=False)
        val_df.drop(columns=ignore_features, inplace=True)

        for col in encoded_columns:
            encoder: OrdinalEncoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

            encoded_col = encoder.fit_transform(train_df[[col]])
            train_df[[col]] = encoded_col

            encoded_col = encoder.transform(val_df[[col]])
            val_df[[col]] = encoded_col

            train_df[col] = train_df[col].astype(int)
            val_df[col] = val_df[col].astype(int)

        model_records_path: str = mytools.get_model_path(f"model_records.json")
        model_records: dict[str, dict] = {}
        if os.path.exists(model_records_path):
            with open(model_records_path, "r", encoding="utf-8") as f:
                model_records = dict(json.load(f))

        if airport not in model_records:
            model_records[airport] = {}

        def _objective(trial):
            _train(
                trial,
                airport,
                train_df.drop(columns=[TARGET_LABEL]),
                val_df.drop(columns=[TARGET_LABEL]),
                train_df[TARGET_LABEL],
                val_df[TARGET_LABEL],
                encoded_columns,
                model_records,
                model_records_path,
            )

        study = optuna.create_study(direction="minimize")
        study.optimize(_objective, n_trials=100)

        print("Number of finished trials: {}".format(len(study.trials)))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: {}".format(trial.value))

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
