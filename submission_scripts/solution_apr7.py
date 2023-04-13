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
from datetime import datetime

def load_model(solution_directory: Path) -> Any:
    """Load any model assets from disk."""
    with (solution_directory / "models.pickle").open("rb") as fp:
        model = pickle.load(fp)
    with (solution_directory / "encoders.pickle").open("rb") as fp:
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

    for col in ["temperature","wind_direction","wind_speed","wind_gust","cloud_ceiling","visibility"]:
        table[col] = prediction["timestamp"].apply(lambda now: lamp.loc[lamp.timestamp <= now].sort_values("timestamp").iloc[-1][col]).fillna(0)
    for col in ["cloud","lightning_prob","precip"]:
        table[col] = prediction["timestamp"].apply(lambda now: lamp.loc[lamp.timestamp <= now].sort_values("timestamp").iloc[-1][col]).fillna("UNK").astype(str)

    table["precip"] = table["precip"].astype(str)

    table["minutes_until_etd"] = minutes_until_etd

    def split_gufi(x: pd.DataFrame):
        information: list = x["gufi"].split(".")
        gufi_flight_number: str = information[0]
        gufi_flight_destination_airport: str = information[2]
        gufi_flight_date: datetime = datetime.strptime(
            "_".join((information[3], information[4], information[5][:2])), "%y%m%d_%H%M_%S"
        )
        gufi_flight_FAA_system: str = information[6]
        gufi_timestamp_until_etd = int((gufi_flight_date - x.timestamp).seconds / 60)
        return pd.Series(
            [
                gufi_flight_number,
                gufi_flight_destination_airport,
                gufi_timestamp_until_etd,
                gufi_flight_date,
                gufi_flight_FAA_system,
            ]
        )

    table[
        [
            "gufi_flight_number",
            "gufi_flight_destination_airport",
            "gufi_timestamp_until_etd",
            "gufi_flight_date",
            "gufi_flight_FAA_system",
        ]
    ] = table.apply(lambda x: split_gufi(x), axis=1)

    table["year"] = table.apply(lambda x: x.timestamp.year, axis=1)
    table["month"] = table.apply(lambda x: x.timestamp.month, axis=1)
    table["day"] = table.apply(lambda x: x.timestamp.day, axis=1)
    table["hour"] = table.apply(lambda x: x.timestamp.hour, axis=1)
    table["minute"] = table.apply(lambda x: x.timestamp.minute, axis=1)
    table["weekday"] = table.apply(lambda x: x.timestamp.weekday(), axis=1)

    table = table.rename(columns={"airport": "airport_x"})

    encoded_columns = ["gufi_flight_destination_airport",
                       "airport_x",
                       "aircraft_engine_class",
                       "aircraft_type",
                       "major_carrier",
                       "flight_type",
                       "cloud",
                       "lightning_prob",
                       "precip"]


    features = ["gufi_flight_destination_airport",
                "month",
                "day",
                "hour",
                "year",
                "weekday",
                "airport_x",
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
    
    #print(table.columns)

    for col in encoded_columns:
        table[[col]] = encoders[col].transform(table[[col]].values)

    for airport in table.airport_x.unique():
        airport_str = str(encoders["airport_x"].inverse_transform(np.asarray([airport]).reshape(-1,1))[0][0])
        #print(type(airport_str))
        #print(airport_str)
        #print(model)
        #print(airport_str in model)
        #print(model[airport_str])
        prediction.loc[table.airport_x == airport, "minutes_until_pushback"] = model[airport_str].predict(table.loc[table.airport_x == airport, features].to_numpy())

    prediction["minutes_until_pushback"] = prediction.minutes_until_pushback.clip(lower=0).fillna(0)

    return prediction