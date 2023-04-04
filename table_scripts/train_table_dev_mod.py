#
# Authors:
# - Kyler Robison
# - Yudong Lin
# - Trevor Tomlin
#
# This script builds a table of training data for a single airport that is hard coded.
#
# It can easily be changed.
#

import multiprocessing
import os
from functools import partial
from typing import Optional
from table_dtype import TableDtype

import feature_engineering
import pandas as pd  # type: ignore
from tqdm import tqdm


# calculate etd
def _process_etd(now: pd.Timestamp, flights_selected: pd.DataFrame, data_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    final_table = flights_selected

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
    delay_3hr = feature_engineering.average_departure_delay(latest_etd, runways, now, 3)
    final_table["delay_3hr"] = pd.Series([delay_3hr] * len(flights_selected), index=flights_selected.index)

    # ----- 30hr Average Delay -----
    delay_30hr = feature_engineering.average_departure_delay(latest_etd, runways, now, 30)
    final_table["delay_30hr"] = pd.Series([delay_30hr] * len(flights_selected), index=flights_selected.index)

    # ----- 3hr Average Time at Stand -----
    standtime_3hr = feature_engineering.average_stand_time(origin, standtimes, now, 3)
    final_table["standtime_3hr"] = pd.Series([standtime_3hr] * len(flights_selected), index=flights_selected.index)

    # ----- 30hr Average Time at Stand -----
    standtime_30hr = feature_engineering.average_stand_time(origin, standtimes, now, 30)
    final_table["standtime_30hr"] = pd.Series([standtime_30hr] * len(flights_selected), index=flights_selected.index)

    return final_table


def _look_for_forecasts(_lamp: pd.DataFrame, _look_for_timestamp: pd.Timestamp, _now: pd.Timestamp) -> Optional[pd.DataFrame]:
    # select all rows contain this forecast_timestamp
    forecasts: pd.DataFrame = _lamp.loc[
        (_lamp.forecast_timestamp == _look_for_timestamp) & (_now - pd.Timedelta(hours=30) <= _lamp.index) & (_lamp.index <= _now)
    ]
    # get the latest forecast
    return forecasts.iloc[forecasts.index.get_indexer([_now], method="nearest")] if forecasts.shape[0] > 0 else None


# add lamp forecast weather information
def _process_lamp(now: pd.Timestamp, flights_selected: pd.DataFrame, _lamp: pd.DataFrame) -> pd.DataFrame:
    # the latest forecast
    latest_forecast: Optional[pd.DataFrame] = None
    # counter to monitoring hours going forward
    hour_f: int = 0 
    # when no valid forecast is found
    while latest_forecast is None:
        # round time to the nearest hour
        forecast_timestamp_look_up = (now - pd.Timedelta(hours=hour_f)).round("H")
        # get the latest forecast
        latest_forecast = _look_for_forecasts(_lamp, forecast_timestamp_look_up, now)
        # if a valid latest forecast is found
        if latest_forecast is not None:
            # then update value
            for key in ("temperature", "wind_direction", "wind_speed", "wind_gust", "cloud_ceiling", "visibility", "cloud", "lightning_prob", "precip"):
                flights_selected[key] = latest_forecast[key].values[0]
            # and break the loop
            break
        # if no forecast within 30 hours can be found
        elif hour_f > 30:
            for key in ("temperature", "wind_direction", "wind_speed", "wind_gust", "cloud_ceiling", "visibility"):
                flights_selected[key] = 0
            for key in ("cloud", "lightning_prob", "precip"):
                flights_selected[key] = "UNK"
            break
        hour_f += 1
    
    return flights_selected


def _process_timestamp(now: pd.Timestamp, flights: pd.DataFrame, data_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    # subset table to only contain flights for the current timestamp
    time_filtered_table: pd.DataFrame = flights.loc[flights.timestamp == now].reset_index(drop=True)
    final_table: pd.DataFrame = _process_etd(now, time_filtered_table, data_tables)
    final_table = _process_lamp(now, final_table, data_tables["lamp"])
    return final_table


def _get_csv_path(*argv: str) -> str:
    etd_csv_path: str = os.path.join(*argv)
    if not os.path.exists(etd_csv_path):
        etd_csv_path += ".bz2"
    return etd_csv_path


def extract_features_for(_df: pd.DataFrame, _airport: str, data_dir: str) -> pd.DataFrame:
    # define list of data tables to load and use for each airport
    feature_tables: dict[str, pd.DataFrame] = {
        "etd": pd.read_csv(_get_csv_path(data_dir, _airport, f"{_airport}_etd.csv"), parse_dates=["departure_runway_estimated_time", "timestamp"]).sort_values(
            "timestamp"
        ),
        "first_position": pd.read_csv(_get_csv_path(data_dir, _airport, f"{_airport}_first_position.csv"), parse_dates=["timestamp"]),
        "lamp": pd.read_csv(_get_csv_path(data_dir, _airport, f"{_airport}_lamp.csv"), parse_dates=["timestamp", "forecast_timestamp"])
        .set_index("timestamp")
        .sort_values("timestamp"),
        "runways": pd.read_csv(_get_csv_path(data_dir, _airport, f"{_airport}_runways.csv"), parse_dates=["departure_runway_actual_time", "timestamp"]),
        "standtimes": pd.read_csv(_get_csv_path(data_dir, _airport, f"{_airport}_standtimes.csv"), parse_dates=["timestamp", "departure_stand_actual_time"]),
    }

    # process all prediction times in parallel
    with multiprocessing.Pool() as executor:
        fn = partial(_process_timestamp, flights=_df, data_tables=feature_tables)
        unique_timestamp = _df.timestamp.unique()
        inputs = zip(pd.to_datetime(unique_timestamp))
        timestamp_tables: list[pd.DataFrame] = executor.starmap(fn, tqdm(inputs, total=len(unique_timestamp)))

    # concatenate individual prediction times to a single dataframe
    _df = pd.concat(timestamp_tables, ignore_index=True)

    # Add runway information
    _df = _df.merge(feature_tables["runways"][["gufi", "departure_runway_actual"]], how="left", on="gufi")

    # Add mfs information
    feature_tables["mfs"] = pd.read_csv(_get_csv_path(data_dir, airport, f"{airport}_mfs.csv"), dtype={"major_carrier": str})
    _df = _df.merge(feature_tables["mfs"], how="left", on="gufi")

    return _df


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

    label_type: str = "prescreened"

    DATA_DIR: str = os.path.join(os.path.dirname(__file__), "..", "_data")

    for airport in airports:
        print("Start processing:", airport)

        # read train labels for given airport
        table: pd.DataFrame = pd.read_csv(
            _get_csv_path(
                DATA_DIR, f"train_labels_{label_type}", f"train_labels_{airport}.csv" if label_type == "open" else f"prescreened_train_labels_{airport}.csv"
            ),
            parse_dates=["timestamp"],
        )
        # table = table.drop_duplicates(subset=["gufi"])

        # extract features for the given airport
        table = extract_features_for(table, airport, DATA_DIR)

        # some int features may be missing due to a lack of information
        table = TableDtype.fix_potential_missing_int_features(table)

        # fill the result missing spot with UNK
        table = table.fillna("UNK")

        # drop isdeparture colum since it is not useful
        table = table.drop(columns=["isdeparture"])

        #Adding global LAMP features
        current = table.copy()

        past_temperatures = (
        current.groupby("timestamp")
        .first()
        .drop(columns=["forecast_timestamp", "time_ahead_prediction"])
        )
        past_temperatures = (
            past_temperatures.rolling("6h").agg({"mean", "min", "max"}).reset_index()
        )
        past_temperatures.columns = [
            "feat_4_" + c[0] + "_" + c[1] + "_last6h"
            if c[0] != "timestamp"
            else "timestamp"
            for c in past_temperatures.columns
        ]
        past_temperatures = (
            past_temperatures.set_index("timestamp")
            .resample("15min")
            .ffill()
            .reset_index()
        )
        
        current_feats = past_temperatures.copy()

        for p in range(1, 24):
            next_temp = (
                current[
                    (current.time_ahead_prediction <= p)
                    & (current.time_ahead_prediction > p - 1)
                ]
                .drop(columns=["forecast_timestamp", "time_ahead_prediction"])
                .groupby("timestamp")
                .mean()
                .reset_index()
            )
            next_temp.columns = [
                "feat_4_" + c + "_next_" + str(p) if c != "timestamp" else "timestamp"
                for c in next_temp.columns
            ]
            next_temp = (
                next_temp.set_index("timestamp").resample("15min").ffill().reset_index()
            )
            current_feats = current_feats.merge(next_temp, how="left", on="timestamp")

        current_feats["airport"] = airport
        weather = pd.DataFrame()

        weather = pd.concat([weather, current_feats])

        table = table.merge(weather, how="left", on=["airport", "timestamp"])

        # Add global weather features
        weather_feats = [c for c in weather.columns if "feat_4" in c]
        for feat in weather_feats:
            table[feat + "_global_min_"] = table["timestamp"].map(
                weather.groupby("timestamp")[feat].min()
            )
            table[feat + "_global_mean"] = table["timestamp"].map(
                weather.groupby("timestamp")[feat].mean()
            )
            table[feat + "_global_max"] = table["timestamp"].map(
                weather.groupby("timestamp")[feat].max()
            )
            table[feat + "_global_std"] = table["timestamp"].map(
                weather.groupby("timestamp")[feat].std()
            )

        # adding feature gufi_end_label since it could be useful
        table["gufi_end_label"] = table.apply(lambda x: "TFM" if x.gufi.endswith("TFM") else "TFM_TFDM" if x.gufi.endswith("TFM_TFDM") else "OTHER", axis=1)

        # table = normalize_str_features(table)

        # save data
        table.to_csv(os.path.join(os.path.dirname(__file__), "..", "train_tables", f"{airport}_full.csv"), index=False)

        print("Finish processing:", airport)
        print("------------------------------")

    print("Done")
