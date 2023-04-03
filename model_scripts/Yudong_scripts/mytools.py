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
from sklearn.preprocessing import OrdinalEncoder  # type: ignore


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


def get_train_tables(_airport: str = "KSEA") -> pd.DataFrame:
    return pd.read_csv(
        os.path.join(os.path.dirname(__file__), "..", "..", "train_tables", f"{_airport}_train.csv"),
        parse_dates=["timestamp"],
        dtype={"minutes_until_etd": int, "minutes_until_pushback": int},
    )


def get_validation_tables(_airport: str = "KSEA") -> pd.DataFrame:
    return pd.read_csv(
        os.path.join(os.path.dirname(__file__), "..", "..", "validation_tables", f"{_airport}_validation.csv"),
        parse_dates=["timestamp"],
        dtype={"minutes_until_etd": int, "minutes_until_pushback": int},
    )


def encodeStr(_data_train: pd.DataFrame, _data_test: pd.DataFrame, col: str) -> None:
    encoder: OrdinalEncoder = OrdinalEncoder()
    _data_train[col] = encoder.fit_transform(_data_train[[col]]) / 100
    _data_test[col] = encoder.transform(_data_test[[col]]) / 100
