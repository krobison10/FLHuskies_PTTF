#
# Author: Kyler Robison
#
# calculate and add traffic information to data table
#

import feature_engineering
import pandas as pd  # type: ignore


# calculate various traffic measures for airport
def add_traffic(
    now: pd.Timestamp, flights_selected: pd.DataFrame, data_tables: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    mfs = data_tables["mfs"]
    etd = feature_engineering.filter_by_timestamp(data_tables["etd"], now, 30)
    runways = feature_engineering.filter_by_timestamp(data_tables["runways"], now, 30)
    standtimes = feature_engineering.filter_by_timestamp(data_tables["standtimes"], now, 30)

    deps_3hr = count_actual_flights(runways, now, 3, departures=True)
    flights_selected["deps_3hr"] = pd.Series([deps_3hr] * len(flights_selected), index=flights_selected.index)

    deps_30hr = count_actual_flights(runways, now, 30, departures=True)
    flights_selected["deps_30hr"] = pd.Series([deps_30hr] * len(flights_selected), index=flights_selected.index)

    arrs_3hr = count_actual_flights(runways, now, 3, departures=False)
    flights_selected["arrs_3hr"] = pd.Series([arrs_3hr] * len(flights_selected), index=flights_selected.index)

    arrs_30hr = count_actual_flights(runways, now, 30, departures=False)
    flights_selected["arrs_30hr"] = pd.Series([arrs_30hr] * len(flights_selected), index=flights_selected.index)

    deps_taxiing = count_planes_taxiing(mfs, runways, standtimes, flights="departures")
    flights_selected["deps_taxiing"] = pd.Series([deps_taxiing] * len(flights_selected), index=flights_selected.index)

    arrs_taxiing = count_planes_taxiing(mfs, runways, standtimes, flights="arrivals")
    flights_selected["arrs_taxiing"] = pd.Series([arrs_taxiing] * len(flights_selected), index=flights_selected.index)

    # apply count of expected departures withing various windows
    flights_selected["exp_deps_15min"] = flights_selected.apply(
        lambda row: count_expected_departures(row["gufi"], etd, 15), axis=1
    )

    flights_selected["exp_deps_30min"] = flights_selected.apply(
        lambda row: count_expected_departures(row["gufi"], etd, 30), axis=1
    )

    # flights_selected["exp_deps_60min"] = flights_selected.apply(
    #     lambda row: count_expected_departures(row["gufi"], etd, 60), axis=1
    # )

    return flights_selected


def count_actual_flights(runways, now, hours: int, departures: bool) -> int:
    runways = feature_engineering.filter_by_timestamp(runways, now, hours)
    if departures:
        runways = runways.loc[pd.notna(runways["departure_runway_actual_time"])]
    else:
        runways = runways.loc[pd.notna(runways["arrival_runway_actual_time"])]

    return runways.shape[0]


def count_planes_taxiing(mfs, runways, standtimes, flights: str) -> int:
    mfs = mfs.loc[mfs["isdeparture"] == (flights == "departures")]

    if flights == "departures":
        taxi = pd.merge(mfs, standtimes, on="gufi")
        taxi = pd.merge(taxi, runways, how="left", on="gufi")
        taxi = taxi.loc[pd.isna(taxi["departure_runway_actual_time"])]
    elif flights == "arrivals":
        taxi = pd.merge(mfs, runways, on="gufi")
        taxi = taxi.loc[pd.notna(taxi["arrival_runway_actual_time"])]
        taxi = pd.merge(taxi, standtimes, how="left", on="gufi")
        taxi = taxi.loc[pd.isna(taxi["arrival_stand_actual_time"])]
    else:
        raise RuntimeError("Invalid argument, must specify departures or arrivals")

    return taxi.shape[0]


def count_expected_departures(gufi: str, etd: pd.DataFrame, window: int) -> int:
    time = etd.loc[etd["gufi"] == gufi]["departure_runway_estimated_time"].iloc[0]

    lower_bound = time - pd.Timedelta(minutes=window)
    upper_bound = time + pd.Timedelta(minutes=window)

    etd_window = etd.loc[
        (etd["departure_runway_estimated_time"] >= lower_bound)
        & (etd["departure_runway_estimated_time"] <= upper_bound)
    ]

    return etd_window.shape[0]


def count_expected_arrivals(data_tables, etd, window):
    pass
