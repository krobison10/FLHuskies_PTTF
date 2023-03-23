#
# Author: Kyler Robison
#
# This script builds a table of training data for a single airport that is hard coded.
# It can easily be changed.
#
# To run on compressed data with format specified in README.md, supply a command line
# argument "compressed" (untested).
#
# This script has some redundant steps as it closely mimics the baseline script.
#

import sys
import multiprocessing
import pandas as pd

from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from functools import partial
from pathlib import Path
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm


def table_for_timestamp(
        now: pd.Timestamp,
        filtered_table: pd.DataFrame,
        etd: pd.DataFrame
) -> pd.DataFrame:
    # subset table to only contain flights for the current timestamp
    time_filtered_table = filtered_table.loc[filtered_table.timestamp == now].reset_index(drop=True)

    # filter features to 30 hours before prediction time to prediction time
    now_etd = etd.loc[(etd.timestamp > now - timedelta(hours=30)) & (etd.timestamp <= now)]

    # Create copy of our mini table
    with_etd = time_filtered_table.copy()

    # get the latest ETD for each flight
    latest_now_etd = now_etd.groupby("gufi").last().departure_runway_estimated_time

    # merge the latest ETD with the flights we are predicting
    departure_runway_estimated_time = time_filtered_table.merge(
        latest_now_etd, how="left", on="gufi"
    ).departure_runway_estimated_time

    with_etd["minutes_until_etd"] = (
        ((departure_runway_estimated_time - time_filtered_table.timestamp).dt.total_seconds() / 60).astype(int)
    )

    return with_etd


if __name__ == "__main__":
    ext = ""
    if len(sys.argv) > 1 and sys.argv[1] == "compressed":
        ext = ".bz2"

    DATA_DIR = Path("../_data")

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

    airport = 'KSEA'

    table = pd.read_csv(DATA_DIR / airport / f"train_labels_{airport}.csv{ext}", parse_dates=["timestamp"])

    # load airport's ETD data and sort by timestamp
    airport_etd = pd.read_csv(
        DATA_DIR / airport / f"features/{airport}_etd.csv{ext}",
        parse_dates=["departure_runway_estimated_time", "timestamp"],
    ).sort_values("timestamp")

    # process all prediction times in parallel
    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        fn = partial(table_for_timestamp, filtered_table=table, etd=airport_etd)
        predictions = list(
            tqdm(
                executor.map(
                    fn,
                    pd.to_datetime(table.timestamp.unique())
                ), total=len(table.timestamp.unique())
            )
        )

    # concatenate individual prediction times to a single dataframe
    predictions = pd.concat(predictions, ignore_index=True)

    # reindex the predictions to match the expected ordering in the submission format
    predictions = (
        predictions.set_index(["gufi", "timestamp", "airport"])
        .loc[table.set_index(["gufi", "timestamp", "airport"]).index].reset_index()
    )

    table = pd.merge(table, predictions.drop(columns=['airport', 'minutes_until_pushback']), on=['gufi', 'timestamp'])

    table.to_csv(Path("../train_tables/etd_only.csv"), index=False)
