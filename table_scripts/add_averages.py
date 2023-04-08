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
    latest_etd: pd.DataFrame = (
        feature_engineering.filter_by_timestamp(data_tables["etd"], now, 30).groupby("gufi").last()
    )
    runways: pd.DataFrame = data_tables["runways"]
    standtimes: pd.DataFrame = data_tables["standtimes"]
    origin: pd.DataFrame = data_tables["first_position"].rename(columns={"timestamp": "origin_time"})

    # ----- 3hr Average Delay -----
    delay_3hr = feature_engineering.average_departure_delay(latest_etd, runways, now, 3)
    flights_selected["delay_3hr"] = pd.Series([delay_3hr] * len(flights_selected), index=flights_selected.index)

    # ----- 30hr Average Delay -----
    delay_30hr = feature_engineering.average_departure_delay(latest_etd, runways, now, 30)
    flights_selected["delay_30hr"] = pd.Series([delay_30hr] * len(flights_selected), index=flights_selected.index)

    # ----- 3hr Average Time at Stand -----
    standtime_3hr = feature_engineering.average_stand_time(origin, standtimes, now, 3)
    flights_selected["standtime_3hr"] = pd.Series([standtime_3hr] * len(flights_selected), index=flights_selected.index)

    # ----- 30hr Average Time at Stand -----
    standtime_30hr = feature_engineering.average_stand_time(origin, standtimes, now, 30)
    flights_selected["standtime_30hr"] = pd.Series(
        [standtime_30hr] * len(flights_selected), index=flights_selected.index
    )

    # WORK IN PROGRESS, NEED TO ENSURE THAT CALCULATION IS FOR DEPARTURES FROM THIS AIRPORT ONLY
    # # ----- 3 hr Average Taxi Time -----
    # taxi_3hr = feature_engineering.average_taxi_time(standtimes, runways, now, 3)
    # flights_selected['avg_taxi_3hr'] = pd.Series([taxi_3hr] * len(flights_selected), index=flights_selected.index)
    #
    # # ----- 30 hr Average Taxi Time -----
    # taxi_30hr = feature_engineering.average_taxi_time(standtimes, runways, now, 30)
    # flights_selected['avg_taxi_30hr'] = pd.Series([taxi_30hr] * len(flights_selected), index=flights_selected.index)

    return flights_selected
