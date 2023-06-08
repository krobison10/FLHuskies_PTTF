"""
Trevor Tomlin
04-17-2023
"""
from functools import partial
import math
import multiprocessing
from pathlib import Path
from typing import Any

from loguru import logger
import pandas as pd
import pickle
import numpy as np
import lightgbm as lgb
from datetime import datetime
from datetime import timedelta
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import re

encoded_columns = [
    # "airport",
    "departure_runways",
    "arrival_runways",
    "cloud",
    "lightning_prob",
    # "precip",
    "gufi_flight_number",
    "gufi_flight_major_carrier",
    "gufi_flight_destination_airport",
    # "gufi_flight_FAA_system",
    # "aircraft_engine_class",
    "aircraft_type",
    "major_carrier",
    "flight_type",
    # "isdeparture"
]


def add_traffic(
    now: pd.Timestamp, flights_selected: pd.DataFrame, data_tables: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    mfs = data_tables["mfs"]
    latest_etd = data_tables["etd"].groupby("gufi", as_index=False).last()
    runways = data_tables["runways"]
    standtimes = data_tables["standtimes"]

    runways_filtered_3hr = filter_by_timestamp(runways, now, 3)

    deps_3hr = count_actual_flights(runways_filtered_3hr, departures=True)
    flights_selected["deps_3hr"] = pd.Series([deps_3hr] * len(flights_selected), index=flights_selected.index)

    deps_30hr = count_actual_flights(runways, departures=True)
    flights_selected["deps_30hr"] = pd.Series([deps_30hr] * len(flights_selected), index=flights_selected.index)

    arrs_3hr = count_actual_flights(runways_filtered_3hr, departures=False)
    flights_selected["arrs_3hr"] = pd.Series([arrs_3hr] * len(flights_selected), index=flights_selected.index)

    arrs_30hr = count_actual_flights(runways, departures=False)
    flights_selected["arrs_30hr"] = pd.Series([arrs_30hr] * len(flights_selected), index=flights_selected.index)

    # technically is the # of planes whom have arrived at destination airport gate and also departed their origin
    # airport over 30 hours ago, but who cares, it's an important feature regardless
    deps_taxiing = count_planes_taxiing(mfs, runways, standtimes, flights="departures")
    flights_selected["deps_taxiing"] = pd.Series([deps_taxiing] * len(flights_selected), index=flights_selected.index)

    arrs_taxiing = count_planes_taxiing(mfs, runways, standtimes, flights="arrivals")
    flights_selected["arrs_taxiing"] = pd.Series([arrs_taxiing] * len(flights_selected), index=flights_selected.index)

    # apply count of expected departures within various windows
    flights_selected["exp_deps_15min"] = flights_selected.apply(
        lambda row: count_expected_departures(row["gufi"], latest_etd, 15), axis=1
    )

    flights_selected["exp_deps_30min"] = flights_selected.apply(
        lambda row: count_expected_departures(row["gufi"], latest_etd, 30), axis=1
    )

    return flights_selected


def count_actual_flights(runways_filtered, departures: bool) -> int:
    if departures:
        runways_filtered = runways_filtered.loc[pd.notna(runways_filtered["departure_runway_actual_time"])]
    else:
        runways_filtered = runways_filtered.loc[pd.notna(runways_filtered["arrival_runway_actual_time"])]

    return runways_filtered.shape[0]


def count_planes_taxiing(mfs, runways, standtimes, flights: str) -> int:
    mfs = mfs.loc[mfs["isdeparture"] == (flights == "departures")]

    if flights == "departures":
        taxi = pd.merge(mfs, standtimes, on="gufi")  # inner join will only result in flights with departure stand times
        taxi = pd.merge(taxi, runways, how="left", on="gufi")  # left join leaves blanks for taxiing flights
        taxi = taxi.loc[pd.isna(taxi["departure_runway_actual_time"])]  # select the taxiing flights
    elif flights == "arrivals":
        taxi = runways.loc[pd.notna(runways["arrival_runway_actual_time"])]  # arrivals are rows with valid time
        taxi = pd.merge(taxi, standtimes, how="left", on="gufi")  # left merge with standtime
        taxi = taxi.loc[pd.isna(taxi["arrival_stand_actual_time"])]  # empty standtimes mean still taxiing
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


def load_model(solution_directory: Path) -> Any:
    """Load any model assets from disk."""
    with (solution_directory / "models.pickle").open("rb") as fp:
        model = pickle.load(fp)
    with (solution_directory / "encoders.pickle").open("rb") as fp:
        encoders = pickle.load(fp)

    return [model, encoders]


def add_date_features(_df: pd.DataFrame) -> pd.DataFrame:
    from pandarallel import pandarallel

    pandarallel.initialize()

    _df["year"] = _df.parallel_apply(lambda x: x.timestamp.year, axis=1)
    _df["month"] = _df.parallel_apply(lambda x: x.timestamp.month, axis=1)
    _df["day"] = _df.parallel_apply(lambda x: x.timestamp.day, axis=1)
    _df["hour"] = _df.parallel_apply(lambda x: x.timestamp.hour, axis=1)
    _df["minute"] = _df.parallel_apply(lambda x: x.timestamp.minute, axis=1)
    _df["weekday"] = _df.parallel_apply(lambda x: x.timestamp.weekday(), axis=1)

    # check if the timestamp given is a holiday
    # us_holidays = holidays.US()
    # _df["is_us_holiday"] = _df.apply(lambda x: x.timestamp in us_holidays, axis=1)

    return _df


def extract_and_add_gufi_features(_df: pd.DataFrame) -> pd.DataFrame:
    from pandarallel import pandarallel

    pandarallel.initialize()

    def _split_gufi(x: pd.DataFrame) -> pd.Series:
        import re
        from datetime import datetime

        information: list = x["gufi"].split(".")
        gufi_flight_number: str = information[0]
        first_int = re.search(r"\d", gufi_flight_number)
        gufi_flight_major_carrier: str = gufi_flight_number[: first_int.start() if first_int is not None else 3]
        gufi_flight_destination_airport: str = information[2]
        gufi_flight_date: datetime = datetime.strptime(
            "_".join((information[3], information[4], information[5][:2])), "%y%m%d_%H%M_%S"
        )
        gufi_flight_FAA_system: str = information[6]
        gufi_timestamp_until_etd = int((gufi_flight_date - x.timestamp).seconds / 60)
        return pd.Series(
            [
                gufi_flight_number,
                gufi_flight_major_carrier,
                gufi_flight_destination_airport,
                gufi_timestamp_until_etd,
                gufi_flight_date,
                gufi_flight_FAA_system,
            ]
        )

    _df[
        [
            "gufi_flight_number",
            "gufi_flight_major_carrier",
            "gufi_flight_destination_airport",
            "gufi_timestamp_until_etd",
            "gufi_flight_date",
            "gufi_flight_FAA_system",
        ]
    ] = _df.parallel_apply(lambda x: _split_gufi(x), axis=1)

    return _df


def average_departure_delay(
    etd_filtered: pd.DataFrame, runways_filtered: pd.DataFrame, column_name: str = "departure_runway_actual_time"
) -> float:
    merged_df = pd.merge(etd_filtered, runways_filtered, on="gufi")

    merged_df["departure_delay"] = (
        merged_df[column_name] - merged_df["departure_runway_estimated_time"]
    ).dt.total_seconds() / 60

    avg_delay: float = merged_df["departure_delay"].mean()
    if math.isnan(avg_delay):
        avg_delay = 0

    return round(avg_delay, 2)


def average_arrival_delay(
    tfm_filtered: pd.DataFrame, runways_filtered: pd.DataFrame, column_name: str = "arrival_runway_actual_time"
) -> float:
    """
    Difference between the time that the airplane was scheduled to arrive and the time it is
    truly arriving
    """
    merged_df = pd.merge(tfm_filtered, runways_filtered, on="gufi")

    merged_df["arrival_delay"] = (
        merged_df[column_name] - merged_df["arrival_runway_estimated_time"]
    ).dt.total_seconds() / 60

    avg_delay: float = merged_df["arrival_delay"].mean()
    if math.isnan(avg_delay):
        avg_delay = 0

    return round(avg_delay, 2)


def average_stand_time(origin_filtered: pd.DataFrame, standtimes_filtered: pd.DataFrame) -> float:
    merged_df = pd.merge(origin_filtered, standtimes_filtered, on="gufi")

    merged_df["avg_stand_time"] = (
        merged_df["origin_time"] - merged_df["departure_stand_actual_time"]
    ).dt.total_seconds() / 60

    avg_stand_time: float = merged_df["avg_stand_time"].mean()
    if math.isnan(avg_stand_time):
        avg_stand_time = 0

    return round(avg_stand_time, 2)


def average_taxi_time(
    mfs: pd.DataFrame, standtimes: pd.DataFrame, runways_filtered: pd.DataFrame, departures: bool = True
) -> float:
    mfs = mfs.loc[mfs["isdeparture"] == departures]

    merged_df = pd.merge(runways_filtered, mfs, on="gufi")
    merged_df = pd.merge(merged_df, standtimes, on="gufi")

    if departures:
        merged_df["taxi_time"] = (
            merged_df["departure_runway_actual_time"] - merged_df["departure_stand_actual_time"]
        ).dt.total_seconds() / 60
    else:
        merged_df["taxi_time"] = (
            merged_df["arrival_stand_actual_time"] - merged_df["arrival_runway_actual_time"]
        ).dt.total_seconds() / 60

    avg_taxi_time: float = merged_df["taxi_time"].mean()
    if math.isnan(avg_taxi_time):
        avg_taxi_time = 0

    return round(avg_taxi_time, 2)


def average_flight_delay(standtimes: pd.DataFrame) -> float:
    """
    Delta between the true time it took to fly to the airport
    and estimated time was supposed to take for the flight to happen
    """
    df = standtimes.copy()

    df["flight_time"] = (df["arrival_stand_actual_time"] - df["departure_stand_actual_time"]).dt.total_seconds() / 60

    avg_flight_time: float = df["flight_time"].mean()
    if math.isnan(avg_flight_time):
        avg_flight_time = 0

    return round(avg_flight_time, 2)


# returns a version of the passed in dataframe that only contains entries
# between the time 'now' and n hours prior
def filter_by_timestamp(df: pd.DataFrame, now: pd.Timestamp, hours: int) -> pd.DataFrame:
    return df.loc[(df.timestamp > now - timedelta(hours=hours)) & (df.timestamp <= now)]


def add_averages(
    now: pd.Timestamp, flights_selected: pd.DataFrame, data_tables: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    mfs: pd.DataFrame = data_tables["mfs"]
    latest_etd: pd.DataFrame = data_tables["etd"].groupby("gufi").last()

    runways: pd.DataFrame = data_tables["runways"]
    standtimes: pd.DataFrame = data_tables["standtimes"]
    origin: pd.DataFrame = data_tables["first_position"].rename(columns={"timestamp": "origin_time"})

    delay_30hr = average_departure_delay(latest_etd, runways)
    flights_selected["delay_30hr"] = pd.Series([delay_30hr] * len(flights_selected), index=flights_selected.index)
    delay_3hr = average_departure_delay(latest_etd, runways)
    flights_selected["delay_3hr"] = pd.Series([delay_3hr] * len(flights_selected), index=flights_selected.index)

    standtime_30hr = average_stand_time(origin, standtimes)
    flights_selected["standtime_30hr"] = pd.Series(
        [standtime_30hr] * len(flights_selected), index=flights_selected.index
    )

    dep_taxi_30hr = average_taxi_time(mfs, standtimes, runways)
    flights_selected["dep_taxi_30hr"] = pd.Series([dep_taxi_30hr] * len(flights_selected), index=flights_selected.index)

    arr_taxi_30hr = average_taxi_time(mfs, standtimes, runways, departures=False)
    flights_selected["arr_taxi_30hr"] = pd.Series([arr_taxi_30hr] * len(flights_selected), index=flights_selected.index)

    # 3 hour features
    latest_etd = filter_by_timestamp(latest_etd, now, 3)
    runways = filter_by_timestamp(runways, now, 3)
    standtimes = filter_by_timestamp(standtimes, now, 3)
    origin = origin.loc[(origin.origin_time > now - timedelta(hours=3)) & (origin.origin_time <= now)]

    standtime_3hr = average_stand_time(origin, standtimes)
    flights_selected["standtime_3hr"] = pd.Series([standtime_3hr] * len(flights_selected), index=flights_selected.index)

    dep_taxi_3hr = average_taxi_time(mfs, standtimes, runways)
    flights_selected["dep_taxi_3hr"] = pd.Series([dep_taxi_3hr] * len(flights_selected), index=flights_selected.index)

    arr_taxi_3hr = average_taxi_time(mfs, standtimes, runways, departures=False)
    flights_selected["arr_taxi_3hr"] = pd.Series([arr_taxi_3hr] * len(flights_selected), index=flights_selected.index)

    latest_etd = filter_by_timestamp(latest_etd, now, 1)
    standtimes = filter_by_timestamp(standtimes, now, 1)
    PDd_1hr = average_departure_delay(latest_etd, standtimes, "departure_stand_actual_time")
    flights_selected["1h_ETDP"] = pd.Series([PDd_1hr] * len(flights_selected), index=flights_selected.index)

    return flights_selected


def add_etd_features(_df: pd.DataFrame, raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts estimated time of departure features and appends it to the existing dataframe
    :param pd.DataFrame _df: Existing feature set at a timestamp-airport level
    :return pd.Dataframe _df: Master table enlarged with additional features
    """

    etd_features = pd.DataFrame()

    etd = raw_data.copy()

    etd["timestamp"] = etd.timestamp.dt.ceil("15min")
    etd["departure_runway_estimated_time"] = pd.to_datetime(etd["departure_runway_estimated_time"])
    etd = etd[etd["timestamp"] < etd["departure_runway_estimated_time"]]

    complete_etd = etd.copy()
    for i in range(1, 4 * 25):
        current = etd.copy()
        current["timestamp"] = current["timestamp"] + pd.Timedelta(f"{i * 15}min")
        current = current[current["timestamp"] < current["departure_runway_estimated_time"]]
        complete_etd = pd.concat([complete_etd, current])

    complete_etd["time_ahead"] = (
        complete_etd["departure_runway_estimated_time"] - complete_etd["timestamp"]
    ).dt.total_seconds()
    complete_etd = complete_etd.groupby(["gufi", "timestamp"]).first().reset_index()

    for i in [30, 60, 180, 1400]:
        complete_etd[f"estdep_next_{i}min"] = (complete_etd["time_ahead"] < i * 60).astype(int)
    complete_etd.sort_values("time_ahead", inplace=True)

    for i in [30, 60, 180, 1400]:
        complete_etd[f"estdep_num_next_{i}min"] = (complete_etd["time_ahead"] < i * 60).astype(int)
    complete_etd.sort_values("time_ahead", inplace=True)

    # number of flights departing from the airport
    etd_aggregation = (
        complete_etd.groupby("timestamp")
        .agg(
            {
                "gufi": "count",
                "estdep_next_30min": "sum",
                "estdep_next_60min": "sum",
                "estdep_next_180min": "sum",
                "estdep_next_1400min": "sum",
            }
        )
        .reset_index()
    )

    etd_aggregation.columns = ["feat_5_" + c if c != "timestamp" else c for c in etd_aggregation.columns]

    etd_features = pd.concat([etd_features, etd_aggregation])

    _df = _df.merge(etd_features, how="left", on=["timestamp"])

    return _df


def add_etd(flights_selected: pd.DataFrame, data_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    final_table = flights_selected

    etd = data_tables["etd"]

    # get the latest ETD for each flight
    latest_etd: pd.DataFrame = etd.groupby("gufi").last()

    # get a series containing latest ETDs for each flight, in the same order they appear in flights
    departure_runway_estimated_time: pd.Series = flights_selected.merge(
        latest_etd.departure_runway_estimated_time, how="left", on="gufi"
    ).departure_runway_estimated_time

    # add new column to flights_selected that represents minutes until pushback
    final_table["minutes_until_etd"] = (
        (departure_runway_estimated_time - flights_selected.timestamp).dt.total_seconds() / 60
    ).astype(int)

    return final_table


def add_config(flights_selected: pd.DataFrame, data_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    # filter features to 30 hours before prediction time to prediction time and save as a copy
    config: pd.DataFrame = data_tables["config"]

    # most recent row of airport configuration
    if config.shape[0] > 0:
        departure_runways = config["departure_runways"].iloc[0]
        arrival_runways = config["arrival_runways"].iloc[0]
    else:
        departure_runways = "UNK"
        arrival_runways = "UNK"

    # add new column for which departure runways are in use at this timestamp
    flights_selected["departure_runways"] = pd.Series(
        [departure_runways] * len(flights_selected), index=flights_selected.index
    )
    flights_selected["arrival_runways"] = pd.Series(
        [arrival_runways] * len(flights_selected), index=flights_selected.index
    )

    return flights_selected


def add_lamp(now: pd.Timestamp, flights_selected: pd.DataFrame, data_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    # forecasts that can be used for prediction
    forecasts_available: pd.DataFrame = data_tables["lamp"]
    # numerical features for lamp
    numerical_feature: tuple[str, ...] = (
        "temperature",
        "wind_direction",
        "wind_speed",
        "wind_gust",
        "cloud_ceiling",
        "visibility",
    )
    # categorical features for lamp
    categorical_feature: tuple[str, ...] = ("cloud", "lightning_prob", "precip")
    # if exists forecasts that can be used for prediction
    if forecasts_available.shape[0] > 0:
        # the latest forecast
        latest_forecast: pd.DataFrame | None = None
        # counter to monitoring hours going forward
        hour_f: int = 0
        # when no valid forecast is found
        while latest_forecast is None:
            # round time to the nearest hour
            forecast_timestamp_to_look_up: pd.Timestamp = (now - pd.Timedelta(hours=hour_f)).round("H")
            # select all rows contain this forecast_timestamp
            forecasts: pd.DataFrame = forecasts_available.loc[
                forecasts_available.forecast_timestamp == forecast_timestamp_to_look_up
            ]
            # get the latest forecast
            latest_forecast = (
                forecasts.iloc[forecasts.index.get_indexer([now], method="nearest")] if forecasts.shape[0] > 0 else None
            )
            # if a valid latest forecast is found
            if latest_forecast is not None:
                # then update value
                for key in numerical_feature + categorical_feature:
                    flights_selected[key] = latest_forecast[key].values[0]
                # and break the loop
                return flights_selected
            # if no forecast within 30 hours can be found
            elif hour_f > 30:
                break
            hour_f += 1
    for key in numerical_feature:
        flights_selected[key] = 0
    for key in categorical_feature:
        flights_selected[key] = "UNK"
    return flights_selected


def _process_timestamp(now: pd.Timestamp, flights: pd.DataFrame, data_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    # subset table to only contain flights for the current timestamp
    filtered_table: pd.DataFrame = flights.loc[flights.timestamp == now].reset_index(drop=True)

    # filters the data tables to only include data from past 30 hours, this call can be omitted in a submission script
    # data_tables = filter_tables(now, data_tables)
    filtered_table = add_etd(filtered_table, data_tables)
    filtered_table = add_traffic(now, filtered_table, data_tables)
    filtered_table = add_averages(now, filtered_table, data_tables)
    filtered_table = add_config(filtered_table, data_tables)
    filtered_table = add_lamp(now, filtered_table, data_tables)

    return filtered_table


def predict(
    config: pd.DataFrame,
    etd: pd.DataFrame,
    first_position: pd.DataFrame,
    lamp: pd.DataFrame,
    mfs: pd.DataFrame,
    runways: pd.DataFrame,
    standtimes: pd.DataFrame,
    tbfm: pd.DataFrame,
    tfm: pd.DataFrame,
    airport: str,
    prediction_time: pd.Timestamp,
    partial_submission_format: pd.DataFrame,
    model: Any,
    solution_directory: Path,
) -> pd.DataFrame:
    """Make predictions for the a set of flights at a single airport and prediction time."""
    logger.debug("Computing prediction based on local models (LGBM) trained on all airports")

    if len(partial_submission_format) == 0:
        return partial_submission_format

    model, encoders = model[0], model[1]

    _df: pd.DataFrame = partial_submission_format.copy()

    feature_tables: dict[str, pd.DataFrame] = {
        "etd": etd.sort_values("timestamp"),
        "config": config.sort_values("timestamp", ascending=False),
        "first_position": first_position,
        "lamp": lamp.set_index("timestamp", drop=False).sort_index(),
        "runways": runways,
        "standtimes": standtimes,
        "mfs": mfs,
    }

    # process all prediction times in parallel
    with multiprocessing.Pool() as executor:
        fn = partial(_process_timestamp, flights=_df, data_tables=feature_tables)
        unique_timestamp = _df.timestamp.unique()
        inputs = zip(pd.to_datetime(unique_timestamp))
        timestamp_tables: list[pd.DataFrame] = executor.starmap(
            fn, tqdm(inputs, total=len(unique_timestamp), disable=True)
        )

    _df = pd.concat(timestamp_tables, ignore_index=True)
    _df = extract_and_add_gufi_features(_df)
    _df = add_date_features(_df)
    _df = add_etd_features(_df, etd)

    _df = _df.merge(mfs[["aircraft_type", "major_carrier", "gufi", "flight_type"]].fillna("UNK"), how="left", on="gufi")

    cat_features = [
        "cloud_ceiling",
        "visibility",
        "year",
        "quarter",
        "month",
        "day",
        "hour",
        "minute",
        "weekday",
    ]

    for col in cat_features:
        _df[col] = _df[col].astype("category")

    features = [
        "airport",
        "minutes_until_etd",
        "deps_3hr",
        "deps_30hr",
        "arrs_3hr",
        "arrs_30hr",
        "deps_taxiing",
        "arrs_taxiing",
        "exp_deps_15min",
        "exp_deps_30min",
        "delay_30hr",
        "standtime_30hr",
        "dep_taxi_30hr",
        "arr_taxi_30hr",
        "delay_3hr",
        "standtime_3hr",
        "dep_taxi_3hr",
        "arr_taxi_3hr",
        "1h_ETDP",
        "departure_runways",
        "arrival_runways",
        "temperature",
        "wind_direction",
        "wind_speed",
        "wind_gust",
        "cloud_ceiling",
        "visibility",
        "cloud",
        "lightning_prob",
        "gufi_flight_number",
        "gufi_flight_major_carrier",
        "gufi_flight_destination_airport",
        "gufi_timestamp_until_etd",
        "gufi_flight_FAA_system",
        "year",
        "quarter",
        "month",
        "day",
        "hour",
        "minute",
        "weekday",
        "feat_5_gufi",
        "feat_5_estdep_next_30min",
        "feat_5_estdep_next_60min",
        "feat_5_estdep_next_180min",
        "feat_5_estdep_next_1400min",
        "aircraft_type",
        "major_carrier",
        "flight_type",
        "isdeparture",
    ]

    prediction = partial_submission_format.copy()

    prediction["minutes_until_pushback"] = model[airport].predict(_df[features], categorical_features="auto")

    prediction["minutes_until_pushback"] = prediction.minutes_until_pushback.clip(lower=0).fillna(0)

    return prediction
