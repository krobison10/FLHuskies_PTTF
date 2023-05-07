#
# Author: Kyler Robison
#
# Dependency for train_table script.
#

import math
from datetime import timedelta, datetime

import pandas as pd  # type: ignore
import polars as pl


def average_departure_delay(
    etd_filtered: pd.DataFrame, runways_filtered: pd.DataFrame, column_name: str = "departure_runway_actual_time"
) -> float:
    merged_df = pd.merge(etd_filtered, runways_filtered, on="gufi")

    merged_df["departure_delay"] = (
        merged_df[column_name] - merged_df["departure_runway_estimated_time"]
    ).dt.total_seconds() / 60

    avg_delay: float = merged_df["departure_delay"].mean()
    if math.isnan(avg_delay):
        avg_delay = 0

    return round(avg_delay, 2)


def average_arrival_delay(
    tfm_filtered: pd.DataFrame, runways_filtered: pd.DataFrame, column_name: str = "arrival_runway_actual_time"
) -> float:
    """
    Difference between the time that the airplane was scheduled to arrive and the time it is
    truly arriving
    """
    merged_df = pd.merge(tfm_filtered, runways_filtered, on="gufi")

    merged_df["arrival_delay"] = (
        merged_df[column_name] - merged_df["arrival_runway_estimated_time"]
    ).dt.total_seconds() / 60

    avg_delay: float = merged_df["arrival_delay"].mean()
    if math.isnan(avg_delay):
        avg_delay = 0

    return round(avg_delay, 2)


def average_arrival_delay_on_prediction(
    tfm_filtered: pd.DataFrame, tbfm_filtered: pd.DataFrame, column_name: str = "arrival_runway_estimated_time"
) -> float:
    """
    Difference between the time that the airplane was scheduled to arrive and the time it is currently
    estimated to arrive
    """
    merged_df = pd.merge(tfm_filtered, tbfm_filtered, on="gufi")

    merged_df["arrival_delay"] = (
        merged_df[column_name] - merged_df["scheduled_runway_estimated_time"]
    ).dt.total_seconds() / 60

    avg_delay: float = merged_df["arrival_delay"].mean()
    if math.isnan(avg_delay):
        avg_delay = 0

    return round(avg_delay, 2)


def average_stand_time(origin_filtered: pd.DataFrame, standtimes_filtered: pd.DataFrame) -> float:
    merged_df = pd.merge(origin_filtered, standtimes_filtered, on="gufi")

    merged_df["avg_stand_time"] = (
        merged_df["origin_time"] - merged_df["departure_stand_actual_time"]
    ).dt.total_seconds() / 60

    avg_stand_time: float = merged_df["avg_stand_time"].mean()
    if math.isnan(avg_stand_time):
        avg_stand_time = 0

    return round(avg_stand_time, 2)


def average_taxi_time(
    mfs: pd.DataFrame, standtimes: pd.DataFrame, runways_filtered: pd.DataFrame, departures: bool = True
) -> float:
    mfs = mfs.loc[mfs["isdeparture"] == departures]

    merged_df = pd.merge(runways_filtered, mfs, on="gufi")
    merged_df = pd.merge(merged_df, standtimes, on="gufi")

    if departures:
        merged_df["taxi_time"] = (
            merged_df["departure_runway_actual_time"] - merged_df["departure_stand_actual_time"]
        ).dt.total_seconds() / 60
    else:
        merged_df["taxi_time"] = (
            merged_df["arrival_stand_actual_time"] - merged_df["arrival_runway_actual_time"]
        ).dt.total_seconds() / 60

    avg_taxi_time: float = merged_df["taxi_time"].mean()
    if math.isnan(avg_taxi_time):
        avg_taxi_time = 0

    return round(avg_taxi_time, 2)


def average_true_flight_time(standtimes: pd.DataFrame) -> float:
    """
    The true time it takes for the flight to happen, avereged over n filtered hours
    Not available for the predicted flight
    """
    df = standtimes.copy()

    df["flight_time"] = (df["arrival_stand_actual_time"] - df["departure_stand_actual_time"]).dt.total_seconds() / 60

    avg_flight_time: float = df["flight_time"].mean()
    if math.isnan(avg_flight_time):
        avg_flight_time = 0

    return round(avg_flight_time, 2)


def average_flight_delay(standtimes: pd.DataFrame) -> float:
    """
    Delta between the true time it took to fly to the airport
    and estimated time was supposed to take for the flight to happen
    """
    df = standtimes.copy()

    df["flight_time"] = (df["arrival_stand_actual_time"] - df["departure_stand_actual_time"]).dt.total_seconds() / 60

    avg_flight_time: float = df["flight_time"].mean()
    if math.isnan(avg_flight_time):
        avg_flight_time = 0

    return round(avg_flight_time, 2)


# returns a version of the passed in dataframe that only contains entries
# between the time 'now' and n hours prior
def filter_by_timestamp(df: pl.DataFrame, now: datetime, hours: int) -> pl.DataFrame:
    return df.filter((pl.col("timestamp") > now - timedelta(hours=hours)) & (pl.col("timestamp") <= now))
