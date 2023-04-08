#
# Authors:
# - Kyler Robison
# - Yudong Lin
#
# generate the full table for a specific airport
#

import multiprocessing
from functools import partial

import pandas as pd  # type: ignore
from add_config import add_config
from add_date import add_date_features
from add_etd import add_etd
from add_averages import add_averages
from add_lamp import add_lamp
from add_mfs import add_mfs
from extract_gufi_features import extract_and_add_gufi_features
from tqdm import tqdm
from utils import get_csv_path


def _process_timestamp(now: pd.Timestamp, flights: pd.DataFrame, data_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    # subset table to only contain flights for the current timestamp
    filtered_table: pd.DataFrame = flights.loc[flights.timestamp == now].reset_index(drop=True)

    filtered_table = add_etd(now, filtered_table, data_tables)
    filtered_table = add_averages(now, filtered_table, data_tables)  # will increase runtime a bit
    filtered_table = add_config(now, filtered_table, data_tables)
    filtered_table = add_lamp(now, filtered_table, data_tables)

    return filtered_table


def generate_table(_airport: str, data_dir: str, max_rows: int = -1) -> pd.DataFrame:
    # read train labels for given airport
    _df: pd.DataFrame = pd.read_csv(
        get_csv_path(data_dir, f"train_labels_prescreened", f"prescreened_train_labels_{_airport}.csv"),
        parse_dates=["timestamp"],
    )

    # table = table.drop_duplicates(subset=["gufi"])

    # if you want to select only a certain amount of row
    if max_rows > 0:
        _df = _df[:max_rows]

    # define list of data tables to load and use for each airport
    feature_tables: dict[str, pd.DataFrame] = {
        "etd": pd.read_csv(
            get_csv_path(data_dir, _airport, f"{_airport}_etd.csv"),
            parse_dates=["departure_runway_estimated_time", "timestamp"],
        ).sort_values("timestamp"),
        "config": pd.read_csv(
            get_csv_path(data_dir, _airport, f"{_airport}_config.csv"), parse_dates=["timestamp"]
        ).sort_values("timestamp", ascending=False),
        "first_position": pd.read_csv(
            get_csv_path(data_dir, _airport, f"{_airport}_first_position.csv"), parse_dates=["timestamp"]
        ),
        "lamp": pd.read_csv(
            get_csv_path(data_dir, _airport, f"{_airport}_lamp.csv"), parse_dates=["timestamp", "forecast_timestamp"]
        )
        .set_index("timestamp")
        .sort_values("timestamp"),
        "runways": pd.read_csv(
            get_csv_path(data_dir, _airport, f"{_airport}_runways.csv"),
            parse_dates=["departure_runway_actual_time", "timestamp"],
        ),
        "standtimes": pd.read_csv(
            get_csv_path(data_dir, _airport, f"{_airport}_standtimes.csv"),
            parse_dates=["timestamp", "departure_stand_actual_time"],
        ),
    }

    # process all prediction times in parallel
    with multiprocessing.Pool() as executor:
        fn = partial(_process_timestamp, flights=_df, data_tables=feature_tables)
        unique_timestamp = _df.timestamp.unique()
        inputs = zip(pd.to_datetime(unique_timestamp))
        timestamp_tables: list[pd.DataFrame] = executor.starmap(fn, tqdm(inputs, total=len(unique_timestamp)))

    # remove feature tables from cache as it is no longer needed
    del feature_tables

    # concatenate individual prediction times to a single dataframe
    _df = pd.concat(timestamp_tables, ignore_index=True)

    # Add runway information
    # _df = _df.merge(feature_tables["runways"][["gufi", "departure_runway_actual"]], how="left", on="gufi")

    # Add mfs information
    _df = add_mfs(_df, get_csv_path(data_dir, _airport, f"{_airport}_mfs.csv"))

    # extract and add mfs information
    _df = extract_and_add_gufi_features(_df)

    # extract holiday features
    _df = add_date_features(_df)

    return _df
