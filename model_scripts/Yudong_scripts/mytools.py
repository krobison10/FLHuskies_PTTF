#
# Author: Yudong Lin
#
# A set of useful tools
#

import json
import os
import pickle
from copy import deepcopy

import lightgbm  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
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


def _get_tables(_path: str, remove_duplicate_gufi: bool, use_cols: list[str] | None = None) -> pd.DataFrame:
    unknown_dtype: dict = {
        "minutes_until_etd": int,
        "minutes_until_pushback": int,
        "precip": str,
        "gufi_flight_major_carrier": str,
        "arrival_runways": str,
    }
    _df: pd.DataFrame = (
        pd.read_csv(_path, parse_dates=["timestamp"], dtype=unknown_dtype)
        if use_cols is None
        else pd.read_csv(_path, dtype=unknown_dtype, usecols=use_cols)
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


_CATEGORICAL_STR_COLUMNS: list[str] = [
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
    "gufi_flight_major_carrier",
]

ENCODED_STR_COLUMNS: list[str] = deepcopy(_CATEGORICAL_STR_COLUMNS)

CATEGORICAL_INT_COLUMNS: list[str] = [
    "cloud_ceiling",
    "visibility",
    "year",
    "month",
    "day",
    "hour",
    "minute",
    "weekday",
]

_FEATURES_IGNORE: list[str] = [
    "gufi",
    "timestamp",
    "airport",
    "isdeparture",
    "aircraft_engine_class",
    "precip",
    "departure_runways",
    "arrival_runways",
    # "visibility",
    # "flight_type",
]


def get_categorical_columns() -> list[str]:
    return ENCODED_STR_COLUMNS + CATEGORICAL_INT_COLUMNS


def get_clean_categorical_columns() -> list[str]:
    ignore_categorical_features(_FEATURES_IGNORE)
    return ENCODED_STR_COLUMNS + CATEGORICAL_INT_COLUMNS


def get_ignored_features() -> list[str]:
    return _FEATURES_IGNORE


def ignore_categorical_features(features_ignore: list[str]) -> None:
    for _ignore in features_ignore:
        if _ignore in ENCODED_STR_COLUMNS:
            ENCODED_STR_COLUMNS.remove(_ignore)
        if _ignore in CATEGORICAL_INT_COLUMNS:
            CATEGORICAL_INT_COLUMNS.remove(_ignore)


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
    if not os.path.exists(get_model_path("models.pickle")):
        raise FileNotFoundError("The model does not exist!")
    _models: dict[str, lightgbm.Booster] = {}
    with open(get_model_path("models.pickle"), "rb") as handle:
        _models = pickle.load(handle)
    return _models[_airport]


def save_model(_airport: str, _model: lightgbm.Booster) -> None:
    _models: dict[str, lightgbm.Booster] = {}
    if os.path.exists(get_model_path("models.pickle")):
        with open(get_model_path("models.pickle"), "rb") as handle:
            _models = pickle.load(handle)
    _models[_airport] = _model
    # save the model
    with open(get_model_path("models.pickle"), "wb") as handle:
        pickle.dump(_models, handle, protocol=pickle.HIGHEST_PROTOCOL)


# get the train and test dataset
def get_train_and_test_ds(_airport: str, category_features_support: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    # load data
    train_df: pd.DataFrame = get_train_tables(_airport, remove_duplicate_gufi=False)
    val_df: pd.DataFrame = get_validation_tables(_airport, remove_duplicate_gufi=False)

    # load encoder
    _ENCODER: dict[str, OrdinalEncoder] = get_encoder()

    # need to make provisions for handling unknown values
    for col in ENCODED_STR_COLUMNS:
        train_df[[col]] = _ENCODER[col].transform(train_df[[col]])
        train_df[col] = train_df[col].astype(int)
        val_df[[col]] = _ENCODER[col].transform(val_df[[col]])
        val_df[col] = val_df[col].astype(int)
    if category_features_support is True:
        for col in get_categorical_columns():
            train_df[col] = train_df[col].astype("category")
            val_df[col] = val_df[col].astype("category")

    # drop useless columns
    train_df.drop(columns=get_ignored_features(), inplace=True)
    val_df.drop(columns=get_ignored_features(), inplace=True)

    return train_df, val_df


class ModelRecords:
    # create or load model records
    __PATH: str = get_model_path(f"model_records.json")
    __DATA: dict[str, dict[str, dict]] = {}
    if os.path.exists(__PATH):
        with open(__PATH, "r", encoding="utf-8") as f:
            __DATA = dict(json.load(f))

    @classmethod
    def get(cls, _airport: str) -> dict[str, dict]:
        if _airport not in cls.__DATA:
            cls.__DATA[_airport] = {}
        return cls.__DATA[_airport]

    @classmethod
    def update(cls, _airport: str, _key: str, _value: dict) -> None:
        cls.get(_airport)[_key] = _value

    @classmethod
    def save(cls) -> None:
        with open(cls.__PATH, "w", encoding="utf-8") as f:
            json.dump(cls.__DATA, f, indent=4, ensure_ascii=False, sort_keys=True)


"""
temp = ["aircraft_type", "major_carrier", "gufi_flight_major_carrier"]
df_t = get_master_tables(use_cols=temp)
with open(get_model_path("report.txt"), "w", encoding="utf-8") as f:
    pd.set_option("display.max_rows", None)
    for _col_t in temp:
        print(df_t[_col_t].value_counts())
        f.write(str(df_t[_col_t].value_counts()))
        print(df_t[_col_t].nunique())
        print("------------------------")
exit()
"""
