#
# Author: Kyler Robison
#
# This script builds a table of training data and a table of validation for a single airport that is hard coded.
# It can easily be changed.
#
# To run on compressed data with format specified in README.md, supply a command line
# argument "compressed".
#

import os
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
    origin: pd.DataFrame = feature_engineering.filter_by_timestamp(data_tables["first_position"], now, 30).copy()

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

    return final_table


if __name__ == "__main__":
    ext = ""
    if len(sys.argv) > 1 and sys.argv[1] == "compressed":
        ext = ".bz2"

    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "_data")

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

    for part in ['train', 'validation']:
        print(f"Generating {part} table for {airport}...")

        labels_path = ""
        output_dir = ""
        if part == "train":
            labels_path = os.path.join(DATA_DIR, "train_labels_prescreened", f"prescreened_train_labels_{airport}.csv{ext}")
            output_dir = os.path.join(os.path.dirname(__file__), "..", "train_tables", f"{airport}_train.csv")
        else:
            labels_path = os.path.join(DATA_DIR, "train_labels_open", f"train_labels_{airport}.csv{ext}")
            output_dir = os.path.join(os.path.dirname(__file__), "..", "validation_tables", f"{airport}_val.csv")

        table: pd.DataFrame = pd.read_csv(labels_path, parse_dates=["timestamp"])
        # table = table.drop_duplicates(subset=["gufi"])

        # define list of data tables to load and use for each airport
        airport_path = os.path.join(DATA_DIR, airport)
        feature_tables: dict[str, pd.DataFrame] = {
            "etd": pd.read_csv(os.path.join(airport_path, f"{airport}_etd.csv{ext}"), parse_dates=["departure_runway_estimated_time", "timestamp"]).sort_values(
                "timestamp"
            ),
            "runways": pd.read_csv(os.path.join(airport_path, f"{airport}_runways.csv{ext}"), parse_dates=["departure_runway_actual_time", "timestamp"]),
            "first_position": pd.read_csv(os.path.join(airport_path, f"{airport}_first_position.csv{ext}"), parse_dates=["timestamp"]),
            "standtimes": pd.read_csv(os.path.join(airport_path, f"{airport}_standtimes.csv{ext}"), parse_dates=["timestamp", "departure_stand_actual_time"]),
        }

        # Add encoded column for runway
        table = table.merge(feature_tables["runways"][["gufi", "departure_runway_actual"]], how="left", on="gufi")
        table["departure_runway_actual"] = table["departure_runway_actual"].fillna("NO_RUNWAY")

        encoder = OrdinalEncoder()
        encoded_runways = encoder.fit_transform(table[["departure_runway_actual"]])
        table["departure_runway"] = encoded_runways
        table["departure_runway"].astype(int)

        # process all prediction times in parallel
        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            fn = partial(process_timestamp, flights=table, data_tables=feature_tables)
            timestamp_tables: list[pd.DataFrame] = list(tqdm(executor.map(fn, pd.to_datetime(table.timestamp.unique())), total=len(table.timestamp.unique())))

        # concatenate individual prediction times to a single dataframe
        table = pd.concat(timestamp_tables, ignore_index=True)

        # move train label column to the end
        cols = table.columns.tolist()
        cols.remove("minutes_until_pushback")
        cols.append("minutes_until_pushback")
        table = table[cols]

        table.to_csv(output_dir, index=False)
