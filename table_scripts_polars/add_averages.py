#
# Author: Kyler Robison
#
# calculate and add traffic information to data table
#

import feature_engineering
import polars as pl
from datetime import datetime, timedelta


# calculate various traffic measures for airport
def add_averages(
    now: datetime, flights_selected: pl.DataFrame, latest_etd: pl.DataFrame, data_tables: dict[str, pl.DataFrame]
) -> pl.DataFrame:
    mfs: pl.DataFrame = data_tables["mfs"]

    runways: pl.DataFrame = data_tables["runways"]
    standtimes: pl.DataFrame = data_tables["standtimes"]
    origin: pl.DataFrame = data_tables["first_position"].rename({"timestamp": "origin_time"})

    # 30 hour features, no need to filter
    delay_30hr = feature_engineering.average_departure_delay(latest_etd, runways)
    stand_time_30hr = feature_engineering.average_stand_time(origin, standtimes)
    dep_taxi_30hr = feature_engineering.average_taxi_time(mfs, standtimes, runways)
    arr_taxi_30hr = feature_engineering.average_taxi_time(mfs, standtimes, runways, departures=False)

    # filter out 3 hour data
    latest_etd = feature_engineering.filter_by_timestamp(latest_etd, now, 3)
    runways = feature_engineering.filter_by_timestamp(runways, now, 3)
    standtimes = feature_engineering.filter_by_timestamp(standtimes, now, 3)
    origin = origin.filter((pl.col("origin_time") > now - timedelta(hours=3)) & (pl.col("origin_time") <= now))

    # obtain 3 hour features
    delay_3hr = feature_engineering.average_departure_delay(latest_etd, runways)
    arr_taxi_3hr = feature_engineering.average_taxi_time(mfs, standtimes, runways, departures=False)
    stand_time_3hr = feature_engineering.average_stand_time(origin, standtimes)
    dep_taxi_3hr = feature_engineering.average_taxi_time(mfs, standtimes, runways)

    # obtain 1 hour features
    latest_etd = feature_engineering.filter_by_timestamp(latest_etd, now, 1)
    standtimes = feature_engineering.filter_by_timestamp(standtimes, now, 1)
    PDd_1hr = feature_engineering.average_departure_delay(latest_etd, standtimes, "departure_stand_actual_time")

    # get the number of flights flights_selected
    flights_selected_len: int = len(flights_selected)

    flights_selected = flights_selected.with_columns(
        [
            pl.lit(delay_30hr * flights_selected_len).alias("delay_30hr"),
            pl.lit(stand_time_30hr * flights_selected_len).alias("standtime_30hr"),
            pl.lit(dep_taxi_30hr * flights_selected_len).alias("dep_taxi_30hr"),
            pl.lit(arr_taxi_30hr * flights_selected_len).alias("arr_taxi_30hr"),
            pl.lit(delay_3hr * flights_selected_len).alias("delay_3hr"),
            pl.lit(stand_time_3hr * flights_selected_len).alias("standtime_3hr"),
            pl.lit(dep_taxi_3hr * flights_selected_len).alias("dep_taxi_3hr"),
            pl.lit(arr_taxi_3hr * flights_selected_len).alias("arr_taxi_3hr"),
            pl.lit(PDd_1hr * flights_selected_len).alias("1h_ETDP"),
        ]
    )

    return flights_selected
