#
# Author: Kyler Robison
#
# calculate and add etd information to the data frame
#

import feature_engineering
import pandas as pd  # type: ignore


# calculate etd
def add_etd(now: pd.Timestamp, flights_selected: pd.DataFrame, data_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    final_table = flights_selected

    # filter features to 30 hours before prediction time to prediction time and save as a copy
    etd: pd.DataFrame = feature_engineering.filter_by_timestamp(data_tables["etd"], now, 30)
    origin: pd.DataFrame = feature_engineering.filter_by_timestamp(data_tables["first_position"], now, 30)
    standtimes: pd.DataFrame = feature_engineering.filter_by_timestamp(data_tables["standtimes"], now, 30)
    # runways: pd.DataFrame = data_tables["runways"]

    # rename origin timestamp to origin_time as to not get confused in future joins,
    # because timestamp is the important feature
    origin = origin.rename(columns={"timestamp": "origin_time"})

    # ----- Minutes Until ETD -----
    # get the latest ETD for each flight
    latest_etd: pd.DataFrame = etd.groupby("gufi").last()

    # get a series containing latest ETDs for each flight, in the same order they appear in flights
    departure_runway_estimated_time: pd.Series = flights_selected.merge(
        latest_etd.departure_runway_estimated_time, how="left", on="gufi"
    ).departure_runway_estimated_time

    # add new column to flights_selected that represents minutes until pushback
    final_table["minutes_until_etd"] = ((departure_runway_estimated_time - flights_selected.timestamp).dt.total_seconds() / 60).astype(int)

    # ----- Minutes Since Origin (WIP) -----
    # get a series containing origin time for each flight, in the same order they appear in flights
    # origin_time: pd.Series = flights_selected.merge(
    #     origin, how="left", on="gufi"
    # ).origin_time

    # add new column to flights_selected that represents minutes since origin
    # flights_selected["minutes_since_origin"] = (
    #     ((flights_selected.timestamp - origin_time).dt.total_seconds() / 60).astype(int)
    # )

    # ----- 3hr Average Delay -----
    # delay_3hr = feature_engineering.average_departure_delay(latest_etd, runways, now, 3)
    # final_table["delay_3hr"] = pd.Series([delay_3hr] * len(flights_selected), index=flights_selected.index)

    # ----- 30hr Average Delay -----
    # delay_30hr = feature_engineering.average_departure_delay(latest_etd, runways, now, 30)
    # final_table["delay_30hr"] = pd.Series([delay_30hr] * len(flights_selected), index=flights_selected.index)

    # ----- 3hr Average Time at Stand -----
    standtime_3hr = feature_engineering.average_stand_time(origin, standtimes, now, 3)
    final_table["standtime_3hr"] = pd.Series([standtime_3hr] * len(flights_selected), index=flights_selected.index)

    # ----- 30hr Average Time at Stand -----
    standtime_30hr = feature_engineering.average_stand_time(origin, standtimes, now, 30)
    final_table["standtime_30hr"] = pd.Series([standtime_30hr] * len(flights_selected), index=flights_selected.index)

    return final_table
