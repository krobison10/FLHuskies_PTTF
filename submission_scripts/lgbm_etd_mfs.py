"""
Trevor Tomlin
03-31-2023
LGBM trained on airport, minutes_until_etd, aircraft_engine_class, aircraft_type, major_carrier, flight_type with all airports
"""
from pathlib import Path
from typing import Any

from loguru import logger
import pandas as pd
import pickle
import numpy as np
import lightgbm as lgb

def load_model(solution_directory: Path) -> Any:
    """Load any model assets from disk."""
    with (solution_directory / "lgbm_etd_mfs.pickle").open("rb") as fp:
        model = pickle.load(fp)
    with (solution_directory / "mfs_encoders.pickle").open("rb") as fp:
        encoders = pickle.load(fp)

    return [model, encoders]


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
    logger.debug("Computing prediction based on LGBM trained on all airports")

    model, encoders = model[0], model[1]

    latest_etd = etd.sort_values("timestamp").groupby("gufi").last().departure_runway_estimated_time

    minutes_until_etd = partial_submission_format.merge(
        latest_etd, how="left", on="gufi"
    ).departure_runway_estimated_time
    
    minutes_until_etd = (minutes_until_etd - partial_submission_format.timestamp).dt.total_seconds()/60

    # Empty dataframe gets passed to the function sometimes
    if len(minutes_until_etd) == 0:
        return partial_submission_format

    prediction = partial_submission_format.copy()

    table = prediction.merge(mfs[["aircraft_engine_class", "aircraft_type", "major_carrier", "flight_type", "gufi"]].fillna("UNK"), how="left", on="gufi")
    table["minutes_until_etd"] = minutes_until_etd

    encoded_columns = ["airport", "aircraft_engine_class", "aircraft_type", "major_carrier", "flight_type"]

    for col in encoded_columns:
        table[[col]] = encoders[col].transform(table[[col]].values)

    prediction["minutes_until_pushback"] = model.predict(table[["airport", "minutes_until_etd", "aircraft_engine_class", "aircraft_type", "major_carrier", "flight_type"]].to_numpy())
    prediction["minutes_until_pushback"] = prediction.minutes_until_pushback.clip(lower=0).fillna(0)

    return prediction