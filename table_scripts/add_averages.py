#
# Author: Kyler Robison
#
# calculate and add traffic information to data table
#

import feature_engineering
import pandas as pd  # type: ignore
from datetime import timedelta

# calculate various traffic measures for airport
def add_averages(
    now: pd.Timestamp, flights_selected: pd.DataFrame, data_tables: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    mfs: pd.DataFrame = data_tables["mfs"]

    latest_etd: pd.DataFrame = data_tables["etd"].groupby("gufi").last()

    latest_tfm: pd.DataFrame = data_tables["tfm"].groupby("gufi").last()

    latest_tbfm: pd.DataFrame = data_tables["tbfm"].groupby("gufi").last()

    runways: pd.DataFrame = data_tables["runways"]
    standtimes: pd.DataFrame = data_tables["standtimes"]
    origin: pd.DataFrame = data_tables["first_position"].rename(columns={"timestamp": "origin_time"})

    # # 30 hour features, no need to filter
    # delay_30hr_dep = feature_engineering.average_departure_delay(latest_etd, runways)
    # flights_selected["delay_30hr_dep"] = pd.Series([delay_30hr_dep] * len(flights_selected), index=flights_selected.index)

    # 30 hour features, no need to filter
    delay_30hr_arr = feature_engineering.average_arrival_delay(latest_tfm, runways)
    flights_selected["delay_30hr_arr"] = pd.Series([delay_30hr_arr] * len(flights_selected), index=flights_selected.index)

    delay_30hr_arr_prediction = feature_engineering.average_arrival_delay_on_prediction(latest_tbfm, runways)
    flights_selected["delay_30hr_arr_pred"] = pd.Series([delay_30hr_arr_prediction] * len(flights_selected), index=flights_selected.index)

    # standtime_30hr = feature_engineering.average_stand_time(origin, standtimes)
    # flights_selected["standtime_30hr"] = pd.Series(
    #     [standtime_30hr] * len(flights_selected), index=flights_selected.index
    # )

    # dep_taxi_30hr = feature_engineering.average_taxi_time(mfs, standtimes, runways)
    # flights_selected['dep_taxi_30hr'] = pd.Series([dep_taxi_30hr] * len(flights_selected), index=flights_selected.index)

    # arr_taxi_30hr = feature_engineering.average_taxi_time(mfs, standtimes, runways, departures=False)
    # flights_selected['arr_taxi_30hr'] = pd.Series([arr_taxi_30hr] * len(flights_selected), index=flights_selected.index)

    av_flight_time30hr = feature_engineering.average_true_flight_time(standtimes)
    flights_selected['flight_time_30hr'] = pd.Series([av_flight_time30hr] * len(flights_selected), index=flights_selected.index)

    # 3 hour features
    latest_etd = feature_engineering.filter_by_timestamp(latest_etd, now, 3)
    latest_tfm = feature_engineering.filter_by_timestamp(latest_tfm, now, 3)
    latest_tbfm = feature_engineering.filter_by_timestamp(latest_tfm, now, 3)
    runways = feature_engineering.filter_by_timestamp(runways, now, 3)
    standtimes = feature_engineering.filter_by_timestamp(standtimes, now, 3)
    origin = origin.loc[(origin.origin_time > now - timedelta(hours=3)) & (origin.origin_time <= now)]


    # delay_3hr_dep = feature_engineering.average_departure_delay(latest_etd, runways)
    # flights_selected["delay_3hr_dep"] = pd.Series([delay_3hr_dep] * len(flights_selected), index=flights_selected.index)

    # 30 hour features, no need to filter
    delay_3hr_arr = feature_engineering.average_arrival_delay(latest_tfm, runways)
    flights_selected["delay_3hr_arr"] = pd.Series([delay_3hr_arr] * len(flights_selected), index=flights_selected.index)
   
    delay_3hr_arr_prediction = feature_engineering.average_arrival_delay_on_prediction(latest_tbfm, runways)
    flights_selected["delay_3hr_arr_pred"] = pd.Series([delay_3hr_arr_prediction] * len(flights_selected), index=flights_selected.index)

    # standtime_3hr = feature_engineering.average_stand_time(origin, standtimes)
    # flights_selected["standtime_3hr"] = pd.Series([standtime_3hr] * len(flights_selected), index=flights_selected.index)

    # dep_taxi_3hr = feature_engineering.average_taxi_time(mfs, standtimes, runways)
    # flights_selected['dep_taxi_3hr'] = pd.Series([dep_taxi_3hr] * len(flights_selected), index=flights_selected.index)

    # arr_taxi_3hr = feature_engineering.average_taxi_time(mfs, standtimes, runways, departures=False)
    # flights_selected['arr_taxi_3hr'] = pd.Series([arr_taxi_3hr] * len(flights_selected), index=flights_selected.index)

    av_flight_time3hr = feature_engineering.average_true_flight_time(standtimes)
    flights_selected['flight_time_3hr'] = pd.Series([av_flight_time3hr] * len(flights_selected), index=flights_selected.index)

    return flights_selected
