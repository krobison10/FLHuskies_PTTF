import multiprocessing
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger
from tqdm import tqdm

from .table_scripts.table_generation import (
    add_date_features,
    add_etd_features,
    extract_and_add_gufi_features,
    process_timestamp,
)

encoded_columns: tuple[str, ...] = (
    # "airport",
    "departure_runways",
    "arrival_runways",
    "cloud",
    "lightning_prob",
    # "precip",
    # "gufi_flight_number",
    "gufi_flight_major_carrier",
    "gufi_flight_destination_airport",
    # "gufi_flight_FAA_system",
    # "aircraft_engine_class",
    "aircraft_type",
    "major_carrier",
    "flight_type",
    # "isdeparture"
)

features: tuple[str, ...] = (
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
    "cloud",
    "lightning_prob",
    "gufi_flight_major_carrier",
    "gufi_flight_destination_airport",
    "gufi_timestamp_until_etd",
    "year",
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
    "visibility",
    "flight_type",
)


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
        fn = partial(process_timestamp, flights=_df, data_tables=feature_tables)
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

    _df["precip"] = _df["precip"].astype(str)

    for col in encoded_columns:
        _df[[col]] = encoders[col].transform(_df[[col]].values)

    # print(_df[features].info())

    # A = set(_df.columns.values.tolist())
    # B = set(model[airport].feature_name())
    # # #print("DF Features: ", A)
    # # #print()
    # # #print("Model Features:" , B)
    # # #print()
    # print("In Features, but not Model Features: ", A-B)
    # print()
    # print("In Model Features, but not Features: ",B-A)

    prediction = partial_submission_format.copy()

    prediction["minutes_until_pushback"] = model[airport].predict(_df[features], categorical_features="auto")

    prediction["minutes_until_pushback"] = prediction.minutes_until_pushback.clip(lower=0).fillna(0)

    return prediction
