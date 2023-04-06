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
from add_etd import add_etd
from add_lamp import add_lamp
from add_mfs import add_mfs
from feature_engineering import get_csv_path
from tqdm import tqdm


def _process_timestamp(now: pd.Timestamp, all_flights: pd.DataFrame, data_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    # subset table to only contain flights for the current timestamp
    filtered_table: pd.DataFrame = all_flights.loc[all_flights.timestamp == now].reset_index(drop=True)
    filtered_table = add_etd(now, filtered_table, data_tables)
    filtered_table = add_lamp(now, filtered_table, data_tables)
    return filtered_table


def generate_table_for(_airport: str, from_dir: str) -> pd.DataFrame:
    # read train labels for given airport
    _df: pd.DataFrame = pd.read_csv(
        get_csv_path(from_dir, f"train_labels_prescreened", f"prescreened_train_labels_{_airport}.csv"),
        parse_dates=["timestamp"],
    )
    # table = table.drop_duplicates(subset=["gufi"])

    # define list of data tables to load and use for each airport
    feature_tables: dict[str, pd.DataFrame] = {
        "etd": pd.read_csv(get_csv_path(from_dir, _airport, f"{_airport}_etd.csv"), parse_dates=["departure_runway_estimated_time", "timestamp"]).sort_values(
            "timestamp"
        ),
        "first_position": pd.read_csv(get_csv_path(from_dir, _airport, f"{_airport}_first_position.csv"), parse_dates=["timestamp"]),
        "lamp": pd.read_csv(get_csv_path(from_dir, _airport, f"{_airport}_lamp.csv"), parse_dates=["timestamp", "forecast_timestamp"])
        .set_index("timestamp")
        .sort_values("timestamp"),
        # "runways": pd.read_csv(get_csv_path(from_dir, _airport, f"{_airport}_runways.csv"), parse_dates=["departure_runway_actual_time", "timestamp"]),
        "standtimes": pd.read_csv(get_csv_path(from_dir, _airport, f"{_airport}_standtimes.csv"), parse_dates=["timestamp", "departure_stand_actual_time"]),
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
    # _df = _df.merge(feature_tables["runways"][["gufi", "departure_runway_actual"]], how="left", on="gufi")

    # Add mfs information
    _df = add_mfs(_df, get_csv_path(from_dir, _airport, f"{_airport}_mfs.csv"))

    return _df