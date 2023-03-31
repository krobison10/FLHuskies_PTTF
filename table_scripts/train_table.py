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
import train_test_split

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
    final_table["minutes_until_etd"] = ((departure_runway_estimated_time - time_filtered_table.timestamp).dt.total_seconds() / 60).astype(int)

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

    airports = [
        "KSEA"
    ]

    for airport in airports:
        print(f"Generating for {airport}")
        print("Loading tables...")

        labels_path = os.path.join(DATA_DIR, "train_labels_prescreened", f"prescreened_train_labels_{airport}.csv{ext}")

        table: pd.DataFrame = pd.read_csv(labels_path, parse_dates=["timestamp"])

        # define list of data tables to load and use for each airport
        airport_path = os.path.join(DATA_DIR, airport)
        feature_tables: dict[str, pd.DataFrame] = {
            "etd": pd.read_csv(os.path.join(airport_path, f"{airport}_etd.csv{ext}"), parse_dates=["departure_runway_estimated_time", "timestamp"]).sort_values(
                "timestamp"
            ),
            "runways": pd.read_csv(os.path.join(airport_path, f"{airport}_runways.csv{ext}"), parse_dates=["departure_runway_actual_time", "timestamp"]),
            "standtimes": pd.read_csv(os.path.join(airport_path, f"{airport}_standtimes.csv{ext}"), parse_dates=["timestamp", "departure_stand_actual_time"]),
        }

        # process all prediction times in parallel
        print("Processing...")
        with multiprocessing.Pool() as executor:
            fn = partial(process_timestamp, flights=table, data_tables=feature_tables)
            unique_timestamp = table.timestamp.unique()
            inputs = zip(pd.to_datetime(unique_timestamp))
            timestamp_tables: list[pd.DataFrame] = executor.starmap(fn, tqdm(inputs, total=len(unique_timestamp)))

        # concatenate individual prediction times to a single dataframe
        print("Concatenating timestamp tables...")
        table = pd.concat(timestamp_tables, ignore_index=True)

        # move train label column to the end
        cols = table.columns.tolist()
        cols.remove("minutes_until_pushback")
        cols.append("minutes_until_pushback")
        table = table[cols]

        # save full table
        print("Saving full table...")
        output_dir = os.path.join(os.path.dirname(__file__), "..", "full_tables", f"{airport}_full.csv")
        table.to_csv(output_dir, index=False)

        # call helper function to split tables and save those as well
        print("Splitting and saving train and validation tables...")
        train_test_split.split(table, airport)
