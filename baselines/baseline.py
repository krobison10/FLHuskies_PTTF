#
# Author: Kyler Robison, Yudong Lin
#
# This script is an implementation of the baseline output from the driven data blog
# which can be found at https://drivendata.co/blog/airport-pushback-benchmark
#
# This script outputs a table that can be submitted to the open arena.
# It also serves as a good starting point for building code that processes the data
# concurrently by dividing and conquering by timestamp to avoid redundant filtering.
#
# To run on compressed data with format specified in README.md, supply command line
# argument "compressed".

import multiprocessing
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from functools import partial

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from tqdm import tqdm


def estimate_pushback(now: pd.Timestamp, cur_submission_format: pd.DataFrame, cur_etd: pd.DataFrame) -> pd.DataFrame:
    # subset submission format to the current prediction time
    now_submission_format: pd.DataFrame = cur_submission_format.loc[cur_submission_format.timestamp == now].reset_index(drop=True)

    # filter features to 30 hours before prediction time to prediction time
    now_etd: pd.DataFrame = cur_etd.loc[(cur_etd.timestamp > now - timedelta(hours=30)) & (cur_etd.timestamp <= now)]

    # get the latest ETD for each flight
    latest_now_etd = now_etd.groupby("gufi").last().departure_runway_estimated_time

    # merge the latest ETD with the flights we are predicting
    departure_runway_estimated_time = now_submission_format.merge(
        latest_now_etd, how="left", on="gufi"
    ).departure_runway_estimated_time

    now_prediction = now_submission_format.copy()

    now_prediction["minutes_until_pushback"] = (
        (departure_runway_estimated_time - now_submission_format.timestamp).dt.total_seconds() / 60
    ) - 15

    return now_prediction


if __name__ == "__main__":
    airports: tuple[str, ...] = (
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
    )

    BASELINE_FILE_DIR: str = os.path.dirname(__file__)

    DATA_DIR: str = os.path.join(BASELINE_FILE_DIR, "..", "_data")

    submission_format: pd.DataFrame = pd.read_csv(
        os.path.join(DATA_DIR, f"submission_format.csv"), parse_dates=["timestamp"]
    )

    for airport in airports:
        print(f"Processing {airport}")
        airport_predictions_path: str = os.path.join(
            BASELINE_FILE_DIR, f"baseline_validation_predictions_{airport}.csv"
        )
        if os.path.exists(airport_predictions_path):
            print(f"Predictions for {airport} already exist.")
            continue

        # subset submission format to current airport
        airport_submission_format: pd.DataFrame = submission_format.loc[submission_format.airport == airport]

        # load airport's ETD data and sort by timestamp
        etd_csv_path: str = os.path.join(DATA_DIR, airport, "features", f"{airport}_etd.csv")
        if not os.path.exists(etd_csv_path) or (len(sys.argv) > 1 and sys.argv[1] == "compressed"):
            etd_csv_path += ".bz2"
        etd: pd.DataFrame = pd.read_csv(
            etd_csv_path,
            parse_dates=["departure_runway_estimated_time", "timestamp"],
        ).sort_values("timestamp")

        # process all prediction times in parallel
        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            fn = partial(estimate_pushback, cur_submission_format=airport_submission_format, cur_etd=etd)
            predictions_t: list = list(
                tqdm(
                    executor.map(
                        fn,
                        pd.to_datetime(airport_submission_format.timestamp.unique())
                    ), total=len(airport_submission_format.timestamp.unique())
                )
            )

        # concatenate individual prediction times to a single dataframe
        predictions = pd.concat(predictions_t, ignore_index=True)
        predictions["minutes_until_pushback"] = predictions.minutes_until_pushback.clip(lower=0).astype(int)

        # reindex the predictions to match the expected ordering in the submission format
        predictions = (
            predictions.set_index(["gufi", "timestamp", "airport"])
            .loc[airport_submission_format.set_index(["gufi", "timestamp", "airport"]).index]
            .reset_index()
        )

        # save the predictions for the current airport
        predictions.to_csv(airport_predictions_path, index=False)

    # concatenate together all tables with names "validation_predictions_<airport>.csv"
    predictions = []

    for airport in airports:
        # read each csv and append to array
        airport_predictions_path = os.path.join(
            BASELINE_FILE_DIR, f"baseline_validation_predictions_{airport}.csv"
        )
        predictions.append(pd.read_csv(airport_predictions_path, parse_dates=["timestamp"]))

    # turn array of dataframes into one dataframe and convert minutes until pushback to int
    predictions = pd.concat(predictions, ignore_index=True)
    predictions["minutes_until_pushback"] = predictions.minutes_until_pushback.astype(int)

    with pd.option_context("float_format", "{:.2f}".format):
        print(predictions.minutes_until_pushback.describe())

    # save big dataframe to csv
    predictions.to_csv("baseline_submission_all.csv", index=False)

    # analyze data
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    predictions.minutes_until_pushback.clip(lower=0, upper=200).hist(bins=np.arange(0, 200), ax=ax)
    ax.set_title("Distribution of predicted minutes to pushback")
    ax.set_ylabel("Number of predictions")
    ax.set_xlabel("Minutes to pushback")
    plt.show()
    # fig.savefig("distribution_of_predicted_minutes_to_pushback_baseline.png")

    # verify format
    assert (predictions.columns == submission_format.columns).all()
    assert len(predictions) == len(submission_format)
    assert predictions[["gufi", "timestamp", "airport"]].equals(submission_format[["gufi", "timestamp", "airport"]])
