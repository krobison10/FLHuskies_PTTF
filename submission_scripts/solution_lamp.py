"""
Trevor Tomlin
04-5-2023
LGBM trained on etd, mfs, lamp with all model on each airports
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
    with (solution_directory / "lgbm_etd_mfs_lamp.pickle").open("rb") as fp:
        model = pickle.load(fp)
    with (solution_directory / "mfs_lamp_encoders.pickle").open("rb") as fp:
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

    minutes_until_etd = (minutes_until_etd - partial_submission_format.timestamp).dt.total_seconds() / 60

    # Empty dataframe gets passed to the function sometimes
    if len(minutes_until_etd) == 0:
        return partial_submission_format

    prediction = partial_submission_format.copy()

    table = prediction.merge(
        mfs[["aircraft_engine_class", "aircraft_type", "major_carrier", "flight_type", "gufi"]].fillna("UNK"),
        how="left",
        on="gufi",
    )
    # table = prediction.merge(lamp[["cloud","lightning_prob","precip"]].fillna("UNK"), how="left", on="gufi")
    # table = prediction.merge(lamp[["temperature","wind_direction","wind_speed","wind_gust","cloud_ceiling","visibility"]].fillna(0), how="left", on="gufi")

    # print(lamp.head())

    # latest_lamp = etd.sort_values("timestamp").last().departure_runway_estimated_time

    # print(prediction.iloc[0].timestamp)
    # print(lamp.loc[lamp.timestamp <= prediction.iloc[0].timestamp].sort_values("timestamp"))

    for col in ["temperature", "wind_direction", "wind_speed", "wind_gust", "cloud_ceiling", "visibility"]:
        table[col] = (
            prediction["timestamp"]
            .apply(lambda now: lamp.loc[lamp.timestamp <= now].sort_values("timestamp").iloc[-1][col])
            .fillna(0)
        )
    for col in ["cloud", "lightning_prob", "precip"]:
        table[col] = (
            prediction["timestamp"]
            .apply(lambda now: lamp.loc[lamp.timestamp <= now].sort_values("timestamp").iloc[-1][col])
            .fillna("UNK")
            .astype(str)
        )
    # print(prediction.head())

    table["minutes_until_etd"] = minutes_until_etd

    encoded_columns = [
        "aircraft_engine_class",
        "aircraft_type",
        "major_carrier",
        "flight_type",
        "cloud",
        "lightning_prob",
        "precip",
    ]

    features = [
        "minutes_until_etd",
        "aircraft_engine_class",
        "aircraft_type",
        "major_carrier",
        "flight_type",
        "temperature",
        "wind_direction",
        "wind_speed",
        "wind_gust",
        "cloud_ceiling",
        "visibility",
        "cloud",
        "lightning_prob",
        "precip",
    ]

    for col in encoded_columns:
        table[[col]] = encoders[col].transform(table[[col]].values)

    # prediction["minutes_until_pushback"] = model.predict(table[[features]].to_numpy())

    for airport in table.airport.unique():
        prediction.loc[table.airport == airport, "minutes_until_pushback"] = model[airport].predict(
            table.loc[table.airport == airport, features].to_numpy()
        )

    prediction["minutes_until_pushback"] = prediction.minutes_until_pushback.clip(lower=0).fillna(0)

    return prediction
