#
# Author: Kyler Robison
#
# This script is an implementation of the baseline output from the driven data blog
# which can be found at https://drivendata.co/blog/airport-pushback-benchmark
#
# This script outputs a table that can be submitted to the open arena.
# It also serves as a good starting point for building code that processes the data
# concurrently by dividing and conquering by timestamp to avoid redundant filtering.
#
# To run on compressed data with format specified in README.md, supply command line
# argument "compressed" (untested).
#
import sys
import pandas as pd
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt

from datetime import timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from tqdm import tqdm

ext = ""
if sys.argv[1] == "compressed":
    ext = ".bz2"

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

DATA_DIR = Path("../_data")

submission_format = pd.read_csv(
    DATA_DIR / f"submission_format.csv", parse_dates=["timestamp"]
)


def estimate_pushback(now: pd.Timestamp, cur_submission_format: pd.DataFrame, cur_etd: pd.DataFrame) -> pd.Series:

    # subset submission format to the current prediction time
    now_submission_format = cur_submission_format.loc[cur_submission_format.timestamp == now].reset_index(drop=True)

    # filter features to 30 hours before prediction time to prediction time
    now_etd = cur_etd.loc[(cur_etd.timestamp > now - timedelta(hours=30)) & (cur_etd.timestamp <= now)]

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


for airport in airports:
    print(f"Processing {airport}")
    airport_predictions_path = Path(f"validation_predictions_{airport}.csv")
    if airport_predictions_path.exists():
        print(f"Predictions for {airport} already exist.")
        continue

    # subset submission format to current airport
    airport_submission_format = submission_format.loc[submission_format.airport == airport]

    # load airport's ETD data and sort by timestamp
    etd = pd.read_csv(
        DATA_DIR / airport / f"features/{airport}_etd.csv{ext}",
        parse_dates=["departure_runway_estimated_time", "timestamp"],
    ).sort_values("timestamp")

    # process all prediction times in parallel
    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        fn = partial(estimate_pushback, cur_submission_format=airport_submission_format, cur_etd=etd)
        predictions = list(
            tqdm(
                executor.map(
                    fn,
                    pd.to_datetime(airport_submission_format.timestamp.unique())
                ), total=len(airport_submission_format.timestamp.unique())
            )
        )

    # concatenate individual prediction times to a single dataframe
    predictions = pd.concat(predictions, ignore_index=True)
    predictions["minutes_until_pushback"] = predictions.minutes_until_pushback.clip(
        lower=0
    ).astype(int)

    # reindex the predictions to match the expected ordering in the submission format
    predictions = (
        predictions.set_index(["gufi", "timestamp", "airport"])
        .loc[airport_submission_format.set_index(["gufi", "timestamp", "airport"]).index].reset_index()
    )

    # save the predictions for the current airport
    predictions.to_csv(airport_predictions_path, index=False)


# concatenate together all tables with names "validation_predictions_<airport>.csv"
predictions = []

for airport in airports:
    # read each csv and append to array
    airport_predictions_path = Path(f"baseline_submission_{airport}.csv")
    predictions.append(pd.read_csv(airport_predictions_path, parse_dates=["timestamp"]))

# turn array of dataframes into one dataframe and convert minutes until pushback to int
predictions = pd.concat(predictions, ignore_index=True)
predictions["minutes_until_pushback"] = predictions.minutes_until_pushback.astype(int)

with pd.option_context("float_format", "{:.2f}".format):
    print(predictions.minutes_until_pushback.describe())

# save big dataframe to csv
predictions.to_csv(Path("baseline_submission_all.csv"), index=False)

# analyze data
fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
predictions.minutes_until_pushback.clip(lower=0, upper=200).hist(bins=np.arange(0, 200), ax=ax)
ax.set_title("Distribution of predicted minutes to pushback")
ax.set_ylabel("Number of predictions")
ax.set_xlabel("Minutes to pushback")
_ = plt.show()

# verify format
assert (predictions.columns == submission_format.columns).all()
assert len(predictions) == len(submission_format)
assert predictions[["gufi", "timestamp", "airport"]].equals(
    submission_format[["gufi", "timestamp", "airport"]]
)
