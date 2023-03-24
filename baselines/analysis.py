import glob
import os
import shutil
from typing import Callable

import matplotlib.pyplot as plt  # type: ignore
import pandas  # type: ignore


def _check_minutes_until_pushback_jump(df_in: pandas.DataFrame) -> None:
    for index, row in df_in.iterrows():
        index = int(index)
        if index > 0:
            last_row = df_in.iloc[index - 1]
            if last_row["gufi"] == row["gufi"]:
                if int(last_row["minutes_until_pushback"]) != int(row["minutes_until_pushback"]) + 15:
                    print("hit:", _path, index)


def emptyFolder(_path: str) -> None:
    if os.path.exists(_path):
        shutil.rmtree(_path)
    os.mkdir(_path)


def generateBarGraph(_data: dict, save_to: str, _x_label: str, y_label: str, title: str) -> None:
    plt.clf()
    plt.bar(_data.keys(), _data.values())
    plt.xlabel(_x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(save_to)


def generateDelayLegendGraph(
    delay_percentages: dict[str, float],
    total_num_counter: dict[str, int],
    _key: str,
    save_to: str,
    _airport: str,
) -> None:
    plt.clf()
    plt.bar(
        total_num_counter.keys(),
        total_num_counter.values(),
        label="total number of departure",
    )
    plt.xlabel(_key)
    plt.ylabel("amount")
    axes2 = plt.twinx()
    axes2.plot(
        delay_percentages.keys(),
        delay_percentages.values(),
        label="delay percentage",
        color="C1",
    )
    axes2.set_ylim(0, 0.1)
    axes2.set_ylabel("delay percentage")
    plt.legend(loc="lower right")
    plt.title("Total flight departure every {0} along with delay percentage for {1}".format(_key, _airport))
    plt.savefig(save_to)


def generateGraphBasedOn(
    df: pandas.DataFrame,
    _func: Callable[[pandas.DataFrame, int], pandas.DataFrame],
    whichAirport: str,
    _key: str,
    _range: tuple[int, int],
    _naming_method: Callable[[int], str],
) -> None:
    delay_percentages: dict[str, float] = {}
    total_num_counter: dict[str, int] = {}
    mean_with_pushback_only: dict[str, int] = {}
    abs_mean: dict[str, int] = {}
    for i in range(_range[0], _range[1]):
        df_filtered: pandas.DataFrame = _func(df, i)
        df_filtered_with_pushback: pandas.DataFrame = df_filtered[df_filtered["minutes_until_pushback"] > 0]
        _ky: str = _naming_method(i)
        total_num_counter[_ky] = df_filtered.shape[0]
        delay_percentages[_ky] = df_filtered_with_pushback.shape[0] / df_filtered.shape[0] if df_filtered.shape[0] > 0 else 0
        abs_mean[_ky] = df_filtered["minutes_until_pushback"].mean()
        mean_with_pushback_only[_ky] = df_filtered_with_pushback["minutes_until_pushback"].mean()
    generateDelayLegendGraph(
        delay_percentages,
        total_num_counter,
        _key,
        os.path.join(
            SUMMARY_DIR_PATH,
            "{0}_{1}_frequency.png".format(whichAirport, _key),
        ),
        whichAirport,
    )
    generateBarGraph(
        abs_mean,
        os.path.join(SUMMARY_DIR_PATH, "{0}_{1}_abs_mean.png".format(whichAirport, _key)),
        _key,
        "time (min)",
        "The absolute average delay time every {0} for {1}".format(_key, whichAirport),
    )
    generateBarGraph(
        mean_with_pushback_only,
        os.path.join(
            SUMMARY_DIR_PATH,
            "{0}_{1}_mean_with_pushback_only.png".format(whichAirport, _key),
        ),
        _key,
        "time (min)",
        "The average delay time every {0} only when there is a delay for {1}".format(_key, whichAirport),
    )


BASELINE_FILE_DIR: str = os.path.dirname(__file__)

SUMMARY_DIR_PATH: str = os.path.join(BASELINE_FILE_DIR, "summary")

emptyFolder(SUMMARY_DIR_PATH)

for _path in glob.glob(os.path.join(BASELINE_FILE_DIR, "baseline_validation_predictions_*.csv")):
    _fileName: str = os.path.basename(_path)
    _df: pandas.DataFrame = pandas.read_csv(_path, parse_dates=["timestamp"])
    # _check_minutes_until_pushback_jump(_df)
    _df = _df.drop_duplicates(subset=["gufi"])
    # include no push back time data
    _airport: str = _fileName.removeprefix("baseline_validation_predictions_").removesuffix(".csv")
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
    )
    # get information bar according to month
    generateGraphBasedOn(
        _df,
        lambda _df, index: _df[_df.timestamp.dt.month == index],
        _airport,
        "month",
        (1, 13),
        lambda _month: str(_month),
    )
    # get information bar according to early/late month
    generateGraphBasedOn(
        _df,
        lambda _df, index: _df[(_df.timestamp.dt.month == index // 2) & (_df.timestamp.dt.day <= 15 if index % 2 == 0 else _df.timestamp.dt.day > 15)],
        _airport,
        "month_by_half",
        (2, 26),
        lambda _month: str(_month // 2) + ("fh" if _month % 2 == 0 else "sh"),
    )
