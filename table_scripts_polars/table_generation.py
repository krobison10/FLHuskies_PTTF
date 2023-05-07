#
# Authors:
# - Kyler Robison
# - Yudong Lin
# - Daniil Filienko
# generate the full table for a specific airport
#

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from functools import partial

import feature_engineering
import polars as pl
from add_averages import add_averages
from add_config import add_config
from add_date import add_date_features
from add_etd import add_etd
from add_etd_features import add_etd_features
from add_lamp import add_lamp
from add_traffic import add_traffic
from extract_gufi_features import extract_and_add_gufi_features
from tqdm import tqdm
from utils import get_csv_path


def _process_timestamp(now: datetime, flights: pl.DataFrame, data_tables: dict[str, pl.DataFrame]) -> pl.DataFrame:
    # subset table to only contain flights for the current timestamp
    filtered_table: pl.DataFrame = flights.filter(pl.col("timestamp") == now)

    # filters the data tables to only include data from past 30 hours, this call can be omitted in a submission script
    data_tables = filter_tables(now, data_tables)

    # get the latest ETD for each flight
    latest_etd: pl.DataFrame = data_tables["etd"].groupby("gufi").last()

    # add features
    filtered_table = add_etd(filtered_table, latest_etd)
    # filtered_table = add_averages(now, filtered_table, latest_etd, data_tables)
    # filtered_table = add_traffic(now, filtered_table, latest_etd, data_tables)
    # filtered_table = add_config(filtered_table, data_tables)
    # filtered_table = add_lamp(now, filtered_table, data_tables)

    return filtered_table


def filter_tables(now: datetime, data_tables: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    new_dict = {}

    for key in data_tables:
        if key != "mfs":
            new_dict[key] = feature_engineering.filter_by_timestamp(data_tables[key], now, 30)

    new_dict["mfs"] = filter_mfs(data_tables["mfs"], new_dict["standtimes"])

    return new_dict


def filter_mfs(mfs: pl.DataFrame, stand_times: pl.DataFrame):
    gufis_wanted: pl.Series = stand_times.select(pl.col("gufi")).unique().to_series()
    mfs_filtered: pl.DataFrame = mfs.filter(pl.col("gufi").is_in(gufis_wanted))
    return mfs_filtered


def generate_table(_airport: str, data_dir: str, max_rows: int = -1) -> pl.DataFrame:
    # read train labels for given airport
    _df: pl.DataFrame = pl.read_csv(
        get_csv_path(data_dir, f"train_labels_prescreened", f"prescreened_train_labels_{_airport}.csv"),
        try_parse_dates=True,
        n_rows=max_rows if max_rows > 0 else None,
    )

    # define list of data tables to load and use for each airport
    feature_tables: dict[str, pl.DataFrame] = {
        "etd": pl.read_csv(get_csv_path(data_dir, _airport, f"{_airport}_etd.csv"), try_parse_dates=True).sort(
            "timestamp"
        ),
        "config": pl.read_csv(get_csv_path(data_dir, _airport, f"{_airport}_config.csv"), try_parse_dates=True).sort(
            "timestamp"
        ),
        "first_position": pl.read_csv(
            get_csv_path(data_dir, _airport, f"{_airport}_first_position.csv"), try_parse_dates=True
        ),
        "lamp": pl.read_csv(get_csv_path(data_dir, _airport, f"{_airport}_lamp.csv"), try_parse_dates=True).sort(
            "timestamp"
        ),
        "runways": pl.read_csv(get_csv_path(data_dir, _airport, f"{_airport}_runways.csv"), try_parse_dates=True),
        "standtimes": pl.read_csv(get_csv_path(data_dir, _airport, f"{_airport}_standtimes.csv"), try_parse_dates=True),
        "mfs": pl.read_csv(get_csv_path(data_dir, _airport, f"{_airport}_mfs.csv")),
    }

    # process all prediction times in parallel
    with multiprocessing.Pool() as executor:
        fn = partial(_process_timestamp, flights=_df, data_tables=feature_tables)
        unique_timestamp = _df.select(pl.col("timestamp")).unique().to_series().to_list()
        inputs = zip(unique_timestamp)
        timestamp_tables: list[pl.DataFrame] = executor.starmap(fn, tqdm(inputs, total=len(unique_timestamp)))

    # concatenate individual prediction times to a single dataframe
    _df = pl.concat(timestamp_tables)

    # Add runway information
    # _df = _df.merge(feature_tables["runways"][["gufi", "departure_runway_actual"]], how="left", on="gufi")

    # extract and add mfs information
    # _df = extract_and_add_gufi_features(_df)

    # extract holiday features
    # _df = add_date_features(_df)

    # Add additional etd features
    # _df = add_etd_features(_df, feature_tables["etd"])

    # Add mfs information
    # _df = _df.merge(feature_tables["mfs"], how="left", on="gufi")

    return _df
