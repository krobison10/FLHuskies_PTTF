"""
Trevor Tomlin
03-28-2023
Linear Regression trained on SEA
"""
# import json
from pathlib import Path
from typing import Any

from loguru import logger
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
import numpy as np


def load_model(solution_directory: Path) -> Any:
    """Load any model assets from disk."""
    with (solution_directory / "linreg.pickle").open("rb") as fp:
        model = pickle.load(fp)

    return model


def predict(
    config: pd.DataFrame,
    etd: pd.DataFrame,
    first_position: pd.DataFrame,
    lamp: pd.DataFrame,
    mfs: pd.DataFrame,
    runways: pd.DataFrame,
    standtimes: pd.DataFrame,
    tbfm: pd.DataFrame,
    tfm: pd.DataFrame,
    airport: str,
    prediction_time: pd.Timestamp,
    partial_submission_format: pd.DataFrame,
    model: Any,
    solution_directory: Path,
) -> pd.DataFrame:
    """Make predictions for the a set of flights at a single airport and prediction time."""
    logger.debug("Computing prediction based on Linear Regression trained on SEA")

    latest_etd = etd.sort_values("timestamp").groupby("gufi").last().departure_runway_estimated_time

    minutes_until_etd = partial_submission_format.merge(
        latest_etd, how="left", on="gufi"
    ).departure_runway_estimated_time

    minutes_until_etd = (minutes_until_etd - partial_submission_format.timestamp).dt.total_seconds() / 60

    # Empty dataframe gets passed to the function sometimes
    if len(minutes_until_etd) == 0:
        return partial_submission_format

    prediction = partial_submission_format.copy()

    prediction["minutes_until_pushback"] = model.predict(minutes_until_etd.to_numpy().reshape(-1, 1))
    prediction["minutes_until_pushback"] = prediction.minutes_until_pushback.clip(lower=0).fillna(0)

    return prediction
