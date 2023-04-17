#
# Author: Yudong Lin
#
# A set of useful tools
#

from copy import deepcopy
import os
import pickle

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
from lightgbm import Booster  # type: ignore
from sklearn.feature_selection import SelectKBest, f_regression  # type: ignore
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder  # type: ignore


def plot_loss(history):
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.ylim([0, 10])
    plt.xlabel("Epoch")
    plt.ylabel("Error [MPG]")
    plt.legend()
    plt.grid(True)


def evaluate_numerical_features(_data: pd.DataFrame, features: tuple[str, ...]) -> None:
    X_train = np.asarray([_data[_col] for _col in features])
    X_train = np.reshape(X_train, (X_train.shape[1], X_train.shape[0]))
    y_train = np.asarray(_data["minutes_until_pushback"])

    fs = SelectKBest(score_func=f_regression)
    fit = fs.fit(X_train, y_train)

    df_scores = pd.DataFrame(fit.scores_)
    df_columns = pd.DataFrame(features)
    features_scores = pd.concat([df_columns, df_scores], axis=1)
    features_scores.columns = ["Selected_columns", "Score"]

    print(features_scores.sort_values("Score"))


def _get_tables(_path: str, remove_duplicate_gufi: bool) -> pd.DataFrame:
    _df: pd.DataFrame = pd.read_csv(
        _path,
        parse_dates=["timestamp"],
        dtype={
            "minutes_until_etd": int,
            "minutes_until_pushback": int,
            "precip": str,
            "gufi_flight_major_carrier": str,
            "arrival_runways": str,
            "departure_runways_ratio": float,
            "arrival_runways_ratio": float,
        },
    ).sort_values(["gufi", "timestamp"])
    if remove_duplicate_gufi is True:
        _df = _df.drop_duplicates(subset=["gufi"])
    return _df


def get_train_tables_path(_airport: str = "KSEA") -> str:
    return os.path.join(os.path.dirname(__file__), "..", "..", "train_tables", f"{_airport}_train.csv")


def get_train_tables(_airport: str = "KSEA", remove_duplicate_gufi: bool = True) -> pd.DataFrame:
    return _get_tables(get_train_tables_path(_airport), remove_duplicate_gufi)


def get_preprocessed_train_tables(_airport: str = "KSEA", remove_duplicate_gufi: bool = True) -> pd.DataFrame:
    return _get_tables(get_train_tables_path(_airport).replace(".csv", "_xgboost.csv"), remove_duplicate_gufi)


def get_validation_tables_path(_airport: str = "KSEA") -> str:
    return os.path.join(os.path.dirname(__file__), "..", "..", "validation_tables", f"{_airport}_validation.csv")


def get_validation_tables(_airport: str = "KSEA", remove_duplicate_gufi: bool = True) -> pd.DataFrame:
    return _get_tables(get_validation_tables_path(_airport), remove_duplicate_gufi)


def get_preprocessed_validation_tables(_airport: str = "KSEA", remove_duplicate_gufi: bool = True) -> pd.DataFrame:
    return _get_tables(get_validation_tables_path(_airport).replace(".csv", "_xgboost.csv"), remove_duplicate_gufi)


def _encodeFeatures(
    _data_train: pd.DataFrame, _data_test: pd.DataFrame, cols: tuple[str, ...], encoder: OrdinalEncoder | MinMaxScaler
) -> None:
    _data_full: pd.DataFrame = pd.concat((_data_train, _data_test))
    for _col in cols:
        encoder.fit(_data_full[[_col]])
        _data_train[_col] = encoder.transform(_data_train[[_col]])
        _data_test[_col] = encoder.transform(_data_test[[_col]])


def encodeStrFeatures(_data_train: pd.DataFrame, _data_test: pd.DataFrame, *cols: str) -> None:
    _encodeFeatures(_data_train, _data_test, cols, OrdinalEncoder())


def normalizeNumericalFeatures(_data_train: pd.DataFrame, _data_test: pd.DataFrame, *cols: str) -> None:
    _encodeFeatures(_data_train, _data_test, cols, MinMaxScaler())


def get_model_path(_fileName: str | None) -> str:
    _dir: str = os.path.join(os.path.dirname(__file__), "..", "..", "models", "yudong_models")
    if not os.path.exists(_dir):
        os.mkdir(_dir)
    return os.path.join(_dir, _fileName) if _fileName is not None else _dir


def log_importance(model, low_score_threshold: int = 2000) -> None:
    _div: str = "--------------------------------------------------"
    with open(get_model_path("report.txt"), "a", encoding="utf-8") as f:
        # --- log gain information -----
        feature_importance_table: dict[str, int] = dict(
            zip(model.feature_name(), model.feature_importance("gain").tolist())
        )
        f.write("feature importance table (gain):\n")
        _keys: list[str] = sorted(feature_importance_table, key=lambda k: feature_importance_table[k])
        for key in _keys:
            f.write(f" - {key}: {feature_importance_table[key]}\n")
        f.write(_div + "\n")
        for key in _keys:
            if feature_importance_table[key] <= low_score_threshold:
                msg = f"feature {key} has a low score of {feature_importance_table[key]}"
                print(msg)
                f.write(msg + "\n")
        print(_div)
        f.write(_div + "\n")
        for key in feature_importance_table:
            if key.startswith("feat_lamp_") and feature_importance_table[key] > low_score_threshold:
                msg = f"find useful global lamp feature {key} has a score of {feature_importance_table[key]}"
                print(msg)
                f.write(msg + "\n")
        # ----- log split information -----
        print(_div)
        f.write(_div + "\n")
        feature_importance_table = dict(zip(model.feature_name(), model.feature_importance("split").tolist()))
        f.write("feature importance table (split):\n")
        _keys = sorted(feature_importance_table, key=lambda k: feature_importance_table[k])
        for key in _keys:
            f.write(f" - {key}: {feature_importance_table[key]}\n")
        f.write(_div + "\n")
        for key in _keys:
            if feature_importance_table[key] <= low_score_threshold:
                msg = f"feature {key} has a low score of {feature_importance_table[key]}"
                print(msg)
                f.write(msg + "\n")
        print(_div)
        f.write(_div + "\n")
        for key in feature_importance_table:
            if key.startswith("feat_lamp_") and feature_importance_table[key] > low_score_threshold:
                msg = f"find useful global lamp feature {key} has a score of {feature_importance_table[key]}"
                print(msg)
                f.write(msg + "\n")


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
    # "ALL",
)

_ALL_ENCODED_STR_COLUMNS: list[str] = [
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
    "gufi_flight_number",
]

ENCODED_STR_COLUMNS: list[str] = deepcopy(_ALL_ENCODED_STR_COLUMNS)

CATEGORICAL_INT_COLUMNS: list[str] = [
    "cloud_ceiling",
    "visibility",
    "year",
    "quarter",
    "month",
    "day",
    "hour",
    "minute",
    "weekday",
]

DEFAULT_IGNORE_FEATURES: list[str] = [
    "gufi",
    "timestamp",
    "gufi_flight_date",
    "gufi_flight_number",
    "isdeparture",
    "airport",
]

CUSTOM_IGNORES: list[str] = [
    "gufi_flight_FAA_system",
    "aircraft_engine_class",
    "departure_runways_ratio",
    "arrival_runways_ratio",
    "quarter",
    "precip",
    "visibility",
    "flight_type",
]


def get_categorical_columns() -> list[str]:
    return ENCODED_STR_COLUMNS + CATEGORICAL_INT_COLUMNS


def get_ignored_features() -> list[str]:
    return CUSTOM_IGNORES + DEFAULT_IGNORE_FEATURES


def ignore_categorical_features(features_ignore: list[str]) -> None:
    for _ignore in features_ignore:
        if _ignore in ENCODED_STR_COLUMNS:
            ENCODED_STR_COLUMNS.remove(_ignore)
        if _ignore in CATEGORICAL_INT_COLUMNS:
            CATEGORICAL_INT_COLUMNS.remove(_ignore)


ignore_categorical_features(DEFAULT_IGNORE_FEATURES)
ignore_categorical_features(CUSTOM_IGNORES)


def get_encoder(_airport: str, train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict[str, OrdinalEncoder]:
    _encoders: dict[str, dict[str, OrdinalEncoder]] = {}
    # generate encoders if not exists
    if os.path.exists(get_model_path("encoders.pickle")):
        with open(get_model_path("encoders.pickle"), "rb") as handle:
            _encoders = pickle.load(handle)
    if _airport not in _encoders:
        _encoder: dict[str, OrdinalEncoder] = {}
        print(f"No encoders found for {_airport} found, will generate one right now.")
        _df: pd.DataFrame = pd.concat([train_df, val_df], ignore_index=True)
        # need to make provisions for handling unknown values
        for _col in _ALL_ENCODED_STR_COLUMNS:
            _encoder[_col] = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(_df[[_col]])
        _encoders[_airport] = _encoder
        # save the encoder
        with open(get_model_path("encoders.pickle"), "wb") as handle:
            pickle.dump(_encoders, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print(f"The encoders for {_airport} are found and will be loaded.")
    return _encoders[_airport]


def save_model(_airport: str, _model: Booster) -> None:
    _models: dict[str, Booster] = {}
    if os.path.exists(get_model_path("models.pickle")):
        with open(get_model_path("models.pickle"), "rb") as handle:
            _models = pickle.load(handle)
    _models[_airport] = _model
    # save the encoder
    with open(get_model_path("models.pickle"), "wb") as handle:
        pickle.dump(_models, handle, protocol=pickle.HIGHEST_PROTOCOL)
