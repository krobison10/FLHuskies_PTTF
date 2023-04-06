#
# Author: Kyler Robison
#
# Dependency for train_table script.
#

import math
import os
from datetime import timedelta

import pandas as pd  # type: ignore


def average_departure_delay(etd: pd.DataFrame, runways: pd.DataFrame, now: pd.Timestamp, hours: int) -> float:
    etd_filtered = filter_by_timestamp(etd, now, hours)
    runways_filtered = filter_by_timestamp(runways, now, hours)

    merged_df = pd.merge(etd_filtered, runways_filtered, on="gufi")

    merged_df["departure_delay"] = (merged_df["departure_runway_actual_time"] - merged_df["departure_runway_estimated_time"]).dt.total_seconds() / 60

    avg_delay: float = merged_df["departure_delay"].mean()
    if math.isnan(avg_delay):
        avg_delay = 0

    return round(avg_delay, 2)


def average_stand_time(origin: pd.DataFrame, standtimes: pd.DataFrame, now: pd.Timestamp, hours: int) -> float:
    origin_filtered = origin.loc[(origin.origin_time > now - timedelta(hours=hours)) & (origin.origin_time <= now)]
    standtimes_filtered = filter_by_timestamp(standtimes, now, hours)

    merged_df = pd.merge(origin_filtered, standtimes_filtered, on="gufi")

    merged_df["avg_stand_time"] = (merged_df["origin_time"] - merged_df["departure_stand_actual_time"]).dt.total_seconds() / 60

    avg_stand_time: float = merged_df["avg_stand_time"].mean()
    if math.isnan(avg_stand_time):
        avg_stand_time = 0

    return round(avg_stand_time, 2)


def average_taxi_time(standtimes: pd.DataFrame, runways: pd.DataFrame, now: pd.Timestamp, hours: int) -> float:
    runways_filtered = filter_by_timestamp(runways, now, hours)

    merged_df = pd.merge(runways_filtered, standtimes, on="gufi")

    merged_df["avg_taxi_time"] = (merged_df["departure_runway_actual_time"] - merged_df["departure_stand_actual_time"]).dt.total_seconds() / 60

    avg_taxi_time: float = merged_df["avg_taxi_time"].mean()
    if math.isnan(avg_taxi_time):
        avg_taxi_time = 0

    return round(avg_taxi_time, 2)


# returns a version of the passed in dataframe that only contains entries
# between the time 'now' and n hours prior
def filter_by_timestamp(df: pd.DataFrame, now: pd.Timestamp, hours: int) -> pd.DataFrame:
    return df.loc[(df.timestamp > now - timedelta(hours=hours)) & (df.timestamp <= now)]


# get a valid path for a csv file
# try to return the path for uncompressed csv file first
# if the uncompressed csv does not exists, then return the path for compressed csv file
def get_csv_path(*argv: str) -> str:
    etd_csv_path: str = os.path.join(*argv)
    if not os.path.exists(etd_csv_path):
        etd_csv_path += ".bz2"
    return etd_csv_path
