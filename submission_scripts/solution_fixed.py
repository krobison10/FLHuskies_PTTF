"""
Created by Trevor Tomlin
Fixed by Yudong Lin
04-15-2023
"""
import multiprocessing
import os
import sys
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger
from sklearn.preprocessing import OrdinalEncoder  # type: ignore
from tqdm import tqdm

sys.path.insert(1, os.path.join(os.path.dirname(__file__), "..", "table_scripts"))

from add_averages import add_averages
from add_config import add_config
from add_date import add_date_features
from add_delta import add_delta
from add_etd import add_etd
from add_etd_features import add_etd_features
from add_lamp import add_lamp
from add_traffic import add_traffic
from extract_gufi_features import extract_and_add_gufi_features

sys.path.pop(1)


def _process_timestamp(now: pd.Timestamp, flights: pd.DataFrame, data_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    # subset table to only contain flights for the current timestamp
    filtered_table: pd.DataFrame = flights.loc[flights.timestamp == now].reset_index(drop=True)

    # filters the data tables to only include data from past 30 hours, this call can be omitted in a submission script
    # data_tables = filter_tables(now, data_tables)
    filtered_table = add_etd(filtered_table, data_tables)
    filtered_table = add_traffic(now, filtered_table, data_tables)
    filtered_table = add_averages(now, filtered_table, data_tables)
    filtered_table = add_delta(now, filtered_table, data_tables)
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
    # _df = add_global_lamp(_df, lamp.reset_index(drop=True))
    _df = add_etd_features(_df, etd)

    _df = _df.merge(
        mfs[["aircraft_engine_class", "aircraft_type", "major_carrier", "flight_type", "gufi", "isdeparture"]].fillna(
            "UNK"
        ),
        how="left",
        on="gufi",
    )

    # for col in ["temperature","wind_direction","wind_speed","wind_gust","cloud_ceiling","visibility"]:
    #     _df[col] = _df["timestamp"].apply(lambda now: lamp.sort_values("timestamp").iloc[-1][col]).fillna(0)
    # for col in ["cloud","lightning_prob","precip"]:
    #     _df[col] = _df["timestamp"].apply(lambda now: lamp.sort_values("timestamp").iloc[-1][col]).fillna("UNK").astype(str)

    # latest_etd = etd.sort_values("timestamp").groupby("gufi").last().departure_runway_estimated_time

    # minutes_until_etd = partial_submission_format.merge(
    #     latest_etd, how="left", on="gufi"
    # ).departure_runway_estimated_time

    # minutes_until_etd = (minutes_until_etd - partial_submission_format.timestamp).dt.total_seconds()/60

    # _df["minutes_until_etd"] = minutes_until_etd

    ignore_features = [
        "gufi",
        "timestamp",
        "gufi_flight_date",
        "gufi_flight_number",
        "isdeparture",
        "dep_ratio",
        "arr_ratio",
    ]

    encoded_columns: list[str] = [
        "cloud",
        "lightning_prob",
        "precip",
        "aircraft_engine_class",
        "aircraft_type",
        "major_carrier",
        "flight_type",
        "departure_runways",
        "arrival_runways",
        "gufi_flight_destination_airport",
        "gufi_flight_FAA_system",
        "gufi_flight_major_carrier",
    ]

    features = [
        x
        for x in _df.columns.values.tolist()
        if x not in ignore_features and not str(x).startswith("feat_lamp_") and not str(x).startswith("feats_lamp_")
    ]

    for i in range(len(encoded_columns) - 1, -1, -1):
        if encoded_columns[i] not in features:
            encoded_columns.pop(i)

    for _col in encoded_columns:
        encoder: OrdinalEncoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        _df[[_col]] = encoder.fit_transform(_df[[_col]])

    prediction = partial_submission_format.copy()

    prediction["minutes_until_pushback"] = model[airport].predict(_df[features], categorical_features=encoded_columns)

    prediction["minutes_until_pushback"] = prediction.minutes_until_pushback.clip(lower=0).fillna(0)

    return prediction
