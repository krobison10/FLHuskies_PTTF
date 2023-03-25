import glob
import os

import matplotlib.pyplot as plt  # type: ignore
import pandas  # type: ignore

from utils import emptyFolder, generateGraphBasedOn

PREDICTION_FILES_DIR: str = os.path.join(os.path.dirname(__file__), "..", "baselines")

SUMMARY_DIR_PATH: str = os.path.join(PREDICTION_FILES_DIR, "summary")

emptyFolder(SUMMARY_DIR_PATH)

for _path in glob.glob(os.path.join(PREDICTION_FILES_DIR, "*_predictions_*.csv")):
    _fileName: str = os.path.basename(_path)
    _df: pandas.DataFrame = pandas.read_csv(_path, parse_dates=["timestamp"])
    # _check_minutes_until_pushback_jump(_df)
    _df = _df.drop_duplicates(subset=["gufi"])
    # include no push back time data
    _airport: str = _fileName[_fileName.index("_predictions_") + 13 :].removesuffix(".csv")
    # get general information bar
    plt.clf()
    pandas.value_counts(_df["minutes_until_pushback"], bins=15).sort_index().plot(kind="bar")
    plt.title("The delay time distribution (only when there is a delay >= 0) for {}".format(_airport))
    plt.xlabel("time interval")
    plt.xlabel("numbers of delay fall into that interval")
    plt.savefig(
        os.path.join(SUMMARY_DIR_PATH, "{}_range.png".format(_airport)),
        bbox_inches="tight",
    )
    _quantiles: pandas.Series = _df["minutes_until_pushback"].quantile([0, 0.25, 0.5, 0.75, 1])
    _quantiles.to_csv(
        os.path.join(
            SUMMARY_DIR_PATH,
            "quantiles_of_" + _fileName,
        )
    )
    # get information bar according to day of the week
    generateGraphBasedOn(
        _df,
        lambda _df, index: _df[_df.timestamp.dt.weekday == index],
        _airport,
        "weekday",
        (0, 7),
        lambda _day: ("Mon", "Tue", "Wed", "Thur", "Fri", "Sat", "Sun")[_day],
        SUMMARY_DIR_PATH,
    )
    # get information bar according to month
    generateGraphBasedOn(_df, lambda _df, index: _df[_df.timestamp.dt.month == index], _airport, "month", (1, 13), lambda _month: str(_month), SUMMARY_DIR_PATH)
