#
# Author: Kyler Robison
# Modified by: Trevor Tomlin
#
# This script builds a table of training data for a single airport that is hard coded.
# It can easily be changed.
#
# To run on compressed data with format specified in README.md, supply a command line
# argument "compressed".
#

import sys
import multiprocessing
import pandas as pd  # type: ignore
import feature_engineering

from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import OrdinalEncoder  # type: ignore
from functools import partial
from pathlib import Path
from tqdm import tqdm


def process_timestamp(now: pd.Timestamp, flights: pd.DataFrame, data_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    # subset table to only contain flights for the current timestamp
    time_filtered_table: pd.DataFrame = flights.loc[flights.timestamp == now].reset_index(drop=True)

    final_table = time_filtered_table.copy()

    # filter features to 30 hours before prediction time to prediction time and save as a copy
    etd: pd.DataFrame = feature_engineering.filter_by_timestamp(data_tables["etd"], now, 30).copy()

    # ----- Minutes Until ETD -----
    # get the latest ETD for each flight
    latest_etd: pd.DataFrame = etd.groupby("gufi").last()

    # get a series containing latest ETDs for each flight, in the same order they appear in flights
    departure_runway_estimated_time: pd.Series = time_filtered_table.merge(
        latest_etd.departure_runway_estimated_time, how="left", on="gufi"
    ).departure_runway_estimated_time

    # add new column to time_filtered_table that represents minutes until pushback
    final_table["minutes_until_etd"] = (
        (departure_runway_estimated_time - time_filtered_table.timestamp).dt.total_seconds() / 60
    ).astype(int)

    return final_table


if __name__ == "__main__":
    ext = ".bz2"

    DATA_DIR = Path("data/KSEA/")

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

    table: pd.DataFrame = pd.read_csv(DATA_DIR / f"train_labels_{airport}.csv{ext}", parse_dates=["timestamp"])

    # define list of data tables to load and use for each airport
    feature_tables: dict[str, pd.DataFrame] = {
        "etd": pd.read_csv(
            DATA_DIR / airport / f"{airport}_etd.csv{ext}", parse_dates=["departure_runway_estimated_time", "timestamp"]
        ).sort_values("timestamp"),
        "mfs": pd.read_csv(DATA_DIR / airport / f"{airport}_mfs.csv{ext}"),
    }

    table = table.merge(
        feature_tables["mfs"][
            ["aircraft_engine_class", "aircraft_type", "major_carrier", "flight_type", "gufi"]
        ].fillna("UNK"),
        how="left",
        on="gufi",
    )

    for col in ["aircraft_engine_class", "aircraft_type", "major_carrier", "flight_type"]:
        encoder = OrdinalEncoder()
        encoded_col = encoder.fit_transform(table[[col]])
        table[col] = encoded_col
        table[col].astype(int)

    # process all prediction times in parallel
    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        fn = partial(process_timestamp, flights=table, data_tables=feature_tables)
        timestamp_tables: list[pd.DataFrame] = list(
            tqdm(executor.map(fn, pd.to_datetime(table.timestamp.unique())), total=len(table.timestamp.unique()))
        )

    # concatenate individual prediction times to a single dataframe
    table = pd.concat(timestamp_tables, ignore_index=True)

    # move train label column to the end
    cols = table.columns.tolist()
    cols.remove("minutes_until_pushback")
    cols.append("minutes_until_pushback")
    table = table[cols]

    # save with name "main.csv"
    table.to_csv(Path("etd_w_mfs.csv"), index=False)
