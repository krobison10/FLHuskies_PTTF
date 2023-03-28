#
# Author: Yudong Lin, Kyler Robison
#
# This script builds a table of training data for a single airport that is hard coded.
# It can easily be changed.
#
# To run on compressed data with format specified in README.md, supply a command line
# argument "compressed".
#

import os
import multiprocessing
import pandas as pd  # type: ignore
import feature_engineering

from sklearn.preprocessing import OrdinalEncoder  # type: ignore
from functools import partial
from tqdm import tqdm


def _process_timestamp(now: pd.Timestamp, flights: pd.DataFrame, data_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    # subset table to only contain flights for the current timestamp
    time_filtered_table: pd.DataFrame = flights.loc[flights.timestamp == now].reset_index(drop=True)

    final_table = time_filtered_table

    # filter features to 30 hours before prediction time to prediction time and save as a copy
    etd: pd.DataFrame = feature_engineering.filter_by_timestamp(data_tables["etd"], now, 30)
    origin: pd.DataFrame = feature_engineering.filter_by_timestamp(data_tables["first_position"], now, 30)
    standtimes: pd.DataFrame = feature_engineering.filter_by_timestamp(data_tables["standtimes"], now, 30)
    runways: pd.DataFrame = data_tables["runways"]

    # rename origin timestamp to origin_time as to not get confused in future joins,
    # because timestamp is the important feature
    origin = origin.rename(columns={"timestamp": "origin_time"})

    # ----- Minutes Until ETD -----
    # get the latest ETD for each flight
    latest_etd: pd.DataFrame = etd.groupby("gufi").last()

    # get a series containing latest ETDs for each flight, in the same order they appear in flights
    departure_runway_estimated_time: pd.Series = time_filtered_table.merge(
        latest_etd.departure_runway_estimated_time, how="left", on="gufi"
    ).departure_runway_estimated_time

    # add new column to time_filtered_table that represents minutes until pushback
    final_table["minutes_until_etd"] = ((departure_runway_estimated_time - time_filtered_table.timestamp).dt.total_seconds() / 60).astype(int)

    # ----- Minutes Since Origin (WIP) -----
    # get a series containing origin time for each flight, in the same order they appear in flights
    # origin_time: pd.Series = time_filtered_table.merge(
    #     origin, how="left", on="gufi"
    # ).origin_time

    # add new column to time_filtered_table that represents minutes since origin
    # time_filtered_table["minutes_since_origin"] = (
    #     ((time_filtered_table.timestamp - origin_time).dt.total_seconds() / 60).astype(int)
    # )

    # ----- 3hr Average Delay -----
    delay_3hr = feature_engineering.average_departure_delay(latest_etd, runways, now, 3)
    final_table["delay_3hr"] = pd.Series([delay_3hr] * len(time_filtered_table), index=time_filtered_table.index)

    # ----- 30hr Average Delay -----
    delay_30hr = feature_engineering.average_departure_delay(latest_etd, runways, now, 30)
    final_table["delay_30hr"] = pd.Series([delay_30hr] * len(time_filtered_table), index=time_filtered_table.index)

    # ----- 3hr Average Time at Stand -----
    standtime_3hr = feature_engineering.average_stand_time(origin, standtimes, now, 3)
    final_table["standtime_3hr"] = pd.Series([standtime_3hr] * len(time_filtered_table), index=time_filtered_table.index)

    # ----- 30hr Average Time at Stand -----
    standtime_30hr = feature_engineering.average_stand_time(origin, standtimes, now, 30)
    final_table["standtime_30hr"] = pd.Series([standtime_30hr] * len(time_filtered_table), index=time_filtered_table.index)

    # ----- get forecast weather information -----
    _lamp = data_tables["lamp"]
    # round time to the nearest hour
    forecast_timestamp_look_up = now.ceil("H")
    # select all rows contain this forecast
    forecasts = _lamp.loc[_lamp.forecast_timestamp == forecast_timestamp_look_up]
    # if no forecast found, try to look for time earlier
    hour_f: int = 0
    while forecasts.shape[0] == 0:
        hour_f += 1
        forecast_timestamp_look_up = (now - pd.Timedelta(hours=hour_f)).ceil("H")
        forecasts = _lamp.loc[_lamp.forecast_timestamp == forecast_timestamp_look_up]
        # if no information within 30 days can be found, then throw error
        if hour_f > 420:
            raise Exception(f"Cannot find forecasts for timestamp {now}")
    # get the latest forecast
    latest_forecast = forecasts.iloc[forecasts.index.get_indexer([now], method="nearest")]
    # update value
    for key in ("temperature", "wind_direction", "wind_speed", "wind_gust", "cloud_ceiling", "visibility", "cloud", "lightning_prob", "precip"):
        final_table[key] = latest_forecast[key].values[0]

    return final_table


def _get_csv_path(*argv: str) -> str:
    etd_csv_path: str = os.path.join(*argv)
    if not os.path.exists(etd_csv_path):
        etd_csv_path += ".bz2"
    return etd_csv_path


def generate(_airport: str, save_to: str) -> None:
    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "_data")

    table: pd.DataFrame = pd.read_csv(_get_csv_path(DATA_DIR, f"train_labels_{_airport}.csv"), parse_dates=["timestamp"])
    # table = table.drop_duplicates(subset=["gufi"])

    # define list of data tables to load and use for each airport
    feature_tables: dict[str, pd.DataFrame] = {
        "etd": pd.read_csv(_get_csv_path(DATA_DIR, _airport, f"{_airport}_etd.csv"), parse_dates=["departure_runway_estimated_time", "timestamp"]).sort_values(
            "timestamp"
        ),
        "runways": pd.read_csv(_get_csv_path(DATA_DIR, _airport, f"{_airport}_runways.csv"), parse_dates=["departure_runway_actual_time", "timestamp"]),
        "first_position": pd.read_csv(_get_csv_path(DATA_DIR, _airport, f"{_airport}_first_position.csv"), parse_dates=["timestamp"]),
        "standtimes": pd.read_csv(_get_csv_path(DATA_DIR, _airport, f"{_airport}_standtimes.csv"), parse_dates=["timestamp", "departure_stand_actual_time"]),
        "lamp": pd.read_csv(_get_csv_path(DATA_DIR, _airport, f"{_airport}_lamp.csv"), parse_dates=["timestamp", "forecast_timestamp"])
        .set_index("timestamp")
        .sort_values("timestamp"),
    }

    # Add encoded column for runway
    table = table.merge(feature_tables["runways"][["gufi", "departure_runway_actual"]], how="left", on="gufi")
    table["departure_runway_actual"] = table["departure_runway_actual"].fillna("NO_RUNWAY")
    encoder = OrdinalEncoder()
    encoded_runways = encoder.fit_transform(table[["departure_runway_actual"]])
    table["departure_runway"] = encoded_runways
    table["departure_runway"].astype(int)

    # process all prediction times in parallel
    with multiprocessing.Pool() as executor:
        fn = partial(_process_timestamp, flights=table, data_tables=feature_tables)
        unique_timestamp = table.timestamp.unique()
        inputs = zip(pd.to_datetime(unique_timestamp))
        timestamp_tables: list[pd.DataFrame] = executor.starmap(fn, tqdm(inputs, total=len(unique_timestamp)))

    # concatenate individual prediction times to a single dataframe
    table = pd.concat(timestamp_tables, ignore_index=True)

    # move train label column to the end
    cols = table.columns.tolist()
    cols.remove("minutes_until_pushback")
    cols.append("minutes_until_pushback")
    table = table[cols]

    # save with name "main.csv"
    table.to_csv(save_to, index=False)


if __name__ == "__main__":
    airports = [
        "KATL",
        "KCLT",
        "KDEN",
        "KDFW",
        "KJFK",
        "KMEM",
        "KMIA",
        "KORD",
        "KPHX",
        "KSEA",
    ]

    airport = "KSEA"

    generate(airport, os.path.join(os.path.dirname(__file__), "..", "train_tables", f"main_{airport}.csv"))
