#
# Author: Yudong Lin
#
# A set of useful tools
#

import json
import os
import pickle
from copy import deepcopy
from glob import glob
from typing import Any

import lightgbm  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd
from constants import AIRLINES
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

    features_scores = pd.DataFrame({"Selected_columns": features, "Score": fit.scores_})

    print(features_scores.sort_values("Score"))


def _get_tables(_path: str, remove_duplicate_gufi: bool, use_cols: list[str] | None = None) -> pd.DataFrame:
    unknown_dtype: dict = {"precip": str, "airline": str, "arrival_runways": str, "year": str}
    for _col in _FLOAT32_COLUMNS:
        unknown_dtype[_col] = "float32"
    for _col in _INT16_COLUMNS:
        unknown_dtype[_col] = "int16"
    _df: pd.DataFrame
    if "ALL_" not in _path:
        _df = (
            pd.read_csv(_path, parse_dates=["timestamp"], dtype=unknown_dtype)
            if use_cols is None
            else pd.read_csv(_path, dtype=unknown_dtype, usecols=use_cols)
        )
    else:
        _df = pd.concat(
            [
                (
                    pd.read_csv(each_csv_path, parse_dates=["timestamp"], dtype=unknown_dtype)
                    if use_cols is None
                    else pd.read_csv(each_csv_path, dtype=unknown_dtype, usecols=use_cols)
                )
                for each_csv_path in glob(os.path.join(os.path.dirname(_path), "*.csv"))
            ],
            ignore_index=True,
        )
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


def get_master_tables_path(_airport: str = "ALL") -> str:
    return os.path.join(os.path.dirname(__file__), "..", "..", "full_tables", f"{_airport}_full.csv")


def get_master_tables(
    _airport: str = "ALL", remove_duplicate_gufi: bool = False, use_cols: list[str] | None = None
) -> pd.DataFrame:
    return _get_tables(get_master_tables_path(_airport), remove_duplicate_gufi, use_cols)


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


_INT16_COLUMNS: tuple[str, ...] = (
    "minutes_until_pushback",
    "minutes_until_etd",
    "deps_3hr",
    "deps_30hr",
    "arrs_3hr",
    "arrs_30hr",
    "deps_taxiing",
    "arrs_taxiing",
    "exp_deps_15min",
    "exp_deps_30min",
    "temperature",
    "wind_direction",
    "wind_speed",
    "wind_gust",
    "cloud_ceiling",
    "visibility",
    "gufi_timestamp_until_etd",
    "feat_5_gufi",
    "feat_5_estdep_next_30min",
    "feat_5_estdep_next_60min",
    "feat_5_estdep_next_180min",
    "feat_5_estdep_next_360min",
)

_FLOAT32_COLUMNS: tuple[str, ...] = (
    "delay_30hr",
    "standtime_30hr",
    "dep_taxi_30hr",
    "arr_taxi_30hr",
    "delay_3hr",
    "standtime_3hr",
    "dep_taxi_3hr",
    "arr_taxi_3hr",
    "1h_ETDP",
)

_CATEGORICAL_STR_COLUMNS: list[str] = [
    "airport",
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
    "airline",
    "year",
]

ENCODED_STR_COLUMNS: list[str] = deepcopy(_CATEGORICAL_STR_COLUMNS)

_CATEGORICAL_INT8_COLUMNS: list[str] = [
    "cloud_ceiling",
    "visibility",
    "month",
    "day",
    "hour",
    "minute",
    "weekday",
]

_FEATURES_IGNORE: list[str] = [
    "gufi",
    "timestamp",
    "isdeparture",
    "aircraft_engine_class",
    "precip",
    "departure_runways",
    "arrival_runways",
    "minute"
    # "visibility",
    # "flight_type",
]


def get_categorical_columns() -> list[str]:
    return ENCODED_STR_COLUMNS + _CATEGORICAL_INT8_COLUMNS


def get_clean_categorical_columns() -> list[str]:
    ignore_categorical_features(_FEATURES_IGNORE)
    return ENCODED_STR_COLUMNS + _CATEGORICAL_INT8_COLUMNS


def get_ignored_features() -> list[str]:
    return _FEATURES_IGNORE


def ignore_categorical_features(features_ignore: list[str]) -> None:
    for _ignore in features_ignore:
        if _ignore in ENCODED_STR_COLUMNS:
            ENCODED_STR_COLUMNS.remove(_ignore)
        if _ignore in _CATEGORICAL_INT8_COLUMNS:
            _CATEGORICAL_INT8_COLUMNS.remove(_ignore)


ignore_categorical_features(_FEATURES_IGNORE)


def get_encoder() -> dict[str, OrdinalEncoder]:
    # generate encoders if not exists
    if os.path.exists(get_model_path("encoders.pickle")):
        with open(get_model_path("encoders.pickle"), "rb") as handle:
            return pickle.load(handle)
    else:
        _encoder: dict[str, OrdinalEncoder] = {}
        print(f"No encoder is found, will generate one right now.")
        _df: pd.DataFrame = get_master_tables(use_cols=_CATEGORICAL_STR_COLUMNS)
        # need to make provisions for handling unknown values
        for _col in _CATEGORICAL_STR_COLUMNS:
            if _col == "cloud":
                _encoder[_col] = OrdinalEncoder(
                    categories=[["BK", "CL", "FEW", "OV", "SC"]],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ).fit(_df[[_col]])
            elif _col == "lightning_prob":
                _encoder[_col] = OrdinalEncoder(
                    categories=[["N", "L", "M", "H"]], handle_unknown="use_encoded_value", unknown_value=-1
                ).fit(_df[[_col]])
            elif _col == "aircraft_engine_class":
                _encoder[_col] = OrdinalEncoder(
                    categories=[["OTHER", "PISTON", "TURBO", "JET"]],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ).fit(_df[[_col]])
            else:
                _encoder[_col] = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(_df[[_col]])
        # save the encoder
        with open(get_model_path("encoders.pickle"), "wb") as handle:
            pickle.dump(_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return _encoder


def get_model(_airport: str) -> lightgbm.Booster:
    if not os.path.exists(get_model_path("models.lightgbm.pickle")):
        raise FileNotFoundError("The model does not exist!")
    _models: dict[str, lightgbm.Booster] = {}
    with open(get_model_path("models.lightgbm.pickle"), "rb") as handle:
        _models = pickle.load(handle)
    return _models[_airport]


def save_model(_airport: str, _model: lightgbm.Booster) -> None:
    _models: dict[str, lightgbm.Booster] = {}
    if os.path.exists(get_model_path("models.lightgbm.pickle")):
        with open(get_model_path("models.lightgbm.pickle"), "rb") as handle:
            _models = pickle.load(handle)
    _models[_airport] = _model
    # save the model
    with open(get_model_path("models.lightgbm.pickle"), "wb") as handle:
        pickle.dump(_models, handle, protocol=pickle.HIGHEST_PROTOCOL)


# get the train and test dataset
def get_train_and_test_ds(_airport: str, valid_airlines_only: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    # load data
    train_df: pd.DataFrame = get_train_tables(_airport, remove_duplicate_gufi=False)
    val_df: pd.DataFrame = get_validation_tables(_airport, remove_duplicate_gufi=False)

    if valid_airlines_only is True:
        train_df = train_df.loc[train_df.airline.isin(AIRLINES)]
        val_df = val_df.loc[val_df.airline.isin(AIRLINES)]

    # load encoder
    _ENCODER: dict[str, OrdinalEncoder] = get_encoder()

    # need to make provisions for handling unknown values
    for col in ENCODED_STR_COLUMNS:
        train_df[[col]] = _ENCODER[col].transform(train_df[[col]])
        val_df[[col]] = _ENCODER[col].transform(val_df[[col]])

    for col in get_categorical_columns():
        train_df[col] = train_df[col].astype("int8")
        val_df[col] = val_df[col].astype("int8")

    # drop useless columns
    train_df.drop(columns=get_ignored_features(), inplace=True)
    val_df.drop(columns=get_ignored_features(), inplace=True)

    return train_df, val_df


class ModelRecords:
    # create or load model records
    __PATH: str = get_model_path("model_records.json")
    __DATA: dict[str, dict[str, dict]] = {}
    __init: bool = False

    @classmethod
    def init(cls) -> None:
        if os.path.exists(cls.__PATH):
            with open(cls.__PATH, "r", encoding="utf-8") as f:
                cls.__DATA = dict(json.load(f))
        cls.__init = True

    @classmethod
    def set_name(cls, fileName: str) -> None:
        cls.__PATH = get_model_path(fileName + ".json")
        cls.__init = False

    @classmethod
    def get(cls, _airport: str) -> dict[str, dict]:
        if not cls.__init:
            cls.init()
        if _airport not in cls.__DATA:
            cls.__DATA[_airport] = {}
        return cls.__DATA[_airport]

    @classmethod
    def update(cls, _airport: str, _key: str, _value: Any, save: bool = False) -> None:
        cls.get(_airport)[_key] = _value
        if save is True:
            cls.save()

    @classmethod
    def save(cls) -> None:
        with open(cls.__PATH, "w", encoding="utf-8") as f:
            json.dump(cls.__DATA, f, indent=4, ensure_ascii=False, sort_keys=True)


def plot_history(_airport: str, history: dict[str, list], saveAsFileName: str | None = None) -> None:
    plt.clf()
    plt.plot(history["loss"], label="loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.title(f"validation loss curve for {_airport}")
    if saveAsFileName is not None:
        plt.savefig(get_model_path(saveAsFileName))
