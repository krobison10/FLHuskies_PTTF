#
# Author: Kyler Robison
#
# calculate and add traffic information to data table
#

import feature_engineering
import pandas as pd  # type: ignore


# calculate various traffic measures for airport
def add_averages(
    now: pd.Timestamp, flights_selected: pd.DataFrame, data_tables: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    latest_etd: pd.DataFrame = data_tables["etd"].groupby("gufi").last()
    runways: pd.DataFrame = data_tables["runways"]
    standtimes: pd.DataFrame = data_tables["standtimes"]
    mfs: pd.DataFrame = data_tables["mfs"]

    origin: pd.DataFrame = data_tables["first_position"].rename(columns={"timestamp": "origin_time"})

    # Delays
    delay_3hr = feature_engineering.average_departure_delay(latest_etd, runways, now, 3)
    flights_selected["delay_3hr"] = pd.Series([delay_3hr] * len(flights_selected), index=flights_selected.index)

    delay_30hr = feature_engineering.average_departure_delay(latest_etd, runways, now, 30)
    flights_selected["delay_30hr"] = pd.Series([delay_30hr] * len(flights_selected), index=flights_selected.index)

    # Times at stands
    standtime_3hr = feature_engineering.average_stand_time(origin, standtimes, now, 3)
    flights_selected["standtime_3hr"] = pd.Series([standtime_3hr] * len(flights_selected), index=flights_selected.index)

    standtime_30hr = feature_engineering.average_stand_time(origin, standtimes, now, 30)
    flights_selected["standtime_30hr"] = pd.Series(
        [standtime_30hr] * len(flights_selected), index=flights_selected.index
    )

    # Taxi times
    dep_taxi_3hr = feature_engineering.average_taxi_time(mfs, standtimes, runways, now, 3)
    flights_selected['dep_taxi_3hr'] = pd.Series([dep_taxi_3hr] * len(flights_selected), index=flights_selected.index)

    dep_taxi_30hr = feature_engineering.average_taxi_time(mfs, standtimes, runways, now, 30)
    flights_selected['dep_taxi_30hr'] = pd.Series([dep_taxi_30hr] * len(flights_selected), index=flights_selected.index)

    arr_taxi_3hr = feature_engineering.average_taxi_time(mfs, standtimes, runways, now, 3, departures=False)
    flights_selected['arr_taxi_3hr'] = pd.Series([arr_taxi_3hr] * len(flights_selected), index=flights_selected.index)

    arr_taxi_30hr = feature_engineering.average_taxi_time(mfs, standtimes, runways, now, 30, departures=False)
    flights_selected['arr_taxi_30hr'] = pd.Series([arr_taxi_30hr] * len(flights_selected), index=flights_selected.index)

    return flights_selected
