#
# Author: Yudong Lin
#
# A set of useful tools
#

import os

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


def _get_tables(_path: str, remove_duplicate_gufi: bool) -> pd.DataFrame:
    _df: pd.DataFrame = pd.read_csv(
        _path, parse_dates=["timestamp"], dtype={"minutes_until_etd": int, "minutes_until_pushback": int, "precip": str}
    ).sort_values(["gufi", "timestamp"])
    if remove_duplicate_gufi is True:
        _df = _df.drop_duplicates(subset=["gufi"])
    return applyAdditionalTimeBasedFeatures(_df)


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


def applyAdditionalTimeBasedFeatures(_data: pd.DataFrame) -> pd.DataFrame:
    _data["month"] = _data.apply(lambda x: x.timestamp.month, axis=1)
    _data["day"] = _data.apply(lambda x: x.timestamp.day, axis=1)
    _data["hour"] = _data.apply(lambda x: x.timestamp.hour, axis=1)
    _data["weekday"] = _data.apply(lambda x: x.timestamp.weekday(), axis=1)
    return _data


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


def get_model_path(_fileName: str) -> str:
    return os.path.join(os.path.dirname(__file__), "..", "..", "models", _fileName)
