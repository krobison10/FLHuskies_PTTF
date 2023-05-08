#
# Author: Kyler Robison
#
# calculate and add traffic information to data table
#

import feature_engineering
import polars as pl
from datetime import datetime, timedelta


# calculate various traffic measures for airport
def add_traffic(
    now: datetime, flights_selected: pl.DataFrame, latest_etd: pl.DataFrame, data_tables: dict[str, pl.DataFrame]
) -> pl.DataFrame:
    mfs = data_tables["mfs"]
    runways = data_tables["runways"]
    standtimes = data_tables["standtimes"]

    runways_filtered_3hr = feature_engineering.filter_by_timestamp(runways, now, 3)

    deps_3hr = count_actual_flights(runways_filtered_3hr, departures=True)
    flights_selected["deps_3hr"] = pl.Series([deps_3hr] * len(flights_selected))

    deps_30hr = count_actual_flights(runways, departures=True)
    flights_selected["deps_30hr"] = pl.Series([deps_30hr] * len(flights_selected))

    arrs_3hr = count_actual_flights(runways_filtered_3hr, departures=False)
    flights_selected["arrs_3hr"] = pl.Series([arrs_3hr] * len(flights_selected))

    arrs_30hr = count_actual_flights(runways, departures=False)
    flights_selected["arrs_30hr"] = pl.Series([arrs_30hr] * len(flights_selected))

    # technically is the # of planes whom have arrived at destination airport gate and also departed their origin
    # airport over 30 hours ago, but who cares, it's an important feature regardless
    deps_taxiing = count_planes_taxiing(mfs, runways, standtimes, flights="departures")
    flights_selected["deps_taxiing"] = pl.Series([deps_taxiing] * len(flights_selected))

    arrs_taxiing = count_planes_taxiing(mfs, runways, standtimes, flights="arrivals")
    flights_selected["arrs_taxiing"] = pl.Series([arrs_taxiing] * len(flights_selected))

    # apply count of expected departures within various windows
    flights_selected["exp_deps_15min"] = flights_selected.apply(
        lambda row: count_expected_departures(row["gufi"], latest_etd, 15), axis=1
    )

    flights_selected["exp_deps_30min"] = flights_selected.apply(
        lambda row: count_expected_departures(row["gufi"], latest_etd, 30), axis=1
    )

    return flights_selected


def count_actual_flights(runways_filtered: pl.DataFrame, departures: bool) -> int:
    if departures:
        runways_filtered = runways_filtered.loc[pl.notna(runways_filtered["departure_runway_actual_time"])]
    else:
        runways_filtered = runways_filtered.loc[pl.notna(runways_filtered["arrival_runway_actual_time"])]

    return runways_filtered.shape[0]


def count_planes_taxiing(mfs: pl.DataFrame, runways: pl.DataFrame, standtimes: pl.DataFrame, flights: str) -> int:
    mfs = mfs.loc[mfs["isdeparture"] == (flights == "departures")]

    if flights == "departures":
        taxi = mfs.join(standtimes, on="gufi")  # inner join will only result in flights with departure stand times
        taxi = taxi.join(runways, how="left", on="gufi")  # left join leaves blanks for taxiing flights
        taxi = taxi.loc[pl.isna(taxi["departure_runway_actual_time"])]  # select the taxiing flights
    elif flights == "arrivals":
        taxi = runways.loc[pl.notna(runways["arrival_runway_actual_time"])]  # arrivals are rows with valid time
        taxi = pl.merge(taxi, standtimes, how="left", on="gufi")  # left merge with standtime
        taxi = taxi.loc[pl.isna(taxi["arrival_stand_actual_time"])]  # empty standtimes mean still taxiing
    else:
        raise RuntimeError("Invalid argument, must specify departures or arrivals")

    return taxi.shape[0]


def count_expected_departures(gufi: str, etd: pl.DataFrame, window: int) -> int:
    time = pl.first(etd.filter(pl.col("gufi") == gufi)["departure_runway_estimated_time"])

    lower_bound = time - timedelta(minutes=window)
    upper_bound = time + timedelta(minutes=window)

    etd_window = etd.filter(
        (pl.col("departure_runway_estimated_time") >= lower_bound)
        & (pl.col("departure_runway_estimated_time") <= upper_bound)
    )

    return etd_window.shape[0]
