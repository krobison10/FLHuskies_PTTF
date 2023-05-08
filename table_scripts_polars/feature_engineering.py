#
# Author: Kyler Robison
#
# Dependency for train_table script.
#

from datetime import datetime, timedelta

import polars as pl

from utils import get_average_difference_in_minutes


def average_departure_delay(
    etd_filtered: pl.DataFrame, runways_filtered: pl.DataFrame, column_name: str = "departure_runway_actual_time"
) -> float:
    return get_average_difference_in_minutes(
        etd_filtered, runways_filtered, column_name, "departure_runway_estimated_time"
    )


def average_stand_time(origin_filtered: pl.DataFrame, stand_times_filtered: pl.DataFrame) -> float:
    return get_average_difference_in_minutes(
        origin_filtered, stand_times_filtered, "origin_time", "departure_stand_actual_time"
    )


def average_taxi_time(
    mfs: pl.DataFrame, standtimes: pl.DataFrame, runways_filtered: pl.DataFrame, departures: bool = True
) -> float:
    merged_df: pl.DataFrame = runways_filtered.join(mfs.filter(pl.col("isdeparture") == departures), on="gufi")

    return (
        get_average_difference_in_minutes(
            merged_df, standtimes, "departure_runway_actual_time", "departure_stand_actual_time"
        )
        if departures
        else get_average_difference_in_minutes(
            merged_df, standtimes, "arrival_stand_actual_time", "arrival_runway_actual_time"
        )
    )


# returns a version of the passed in dataframe that only contains entries
# between the time 'now' and n hours prior
def filter_by_timestamp(df: pl.DataFrame, now: datetime, hours: int) -> pl.DataFrame:
    return df.filter((pl.col("timestamp") > now - timedelta(hours=hours)) & (pl.col("timestamp") <= now))
