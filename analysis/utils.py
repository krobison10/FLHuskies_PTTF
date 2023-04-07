#
# Author: Yudong Lin
#
# A set of function that can be used to generate graph
# Use for data analysis
#

import os
import shutil
from typing import Callable
import matplotlib.pyplot as plt  # type: ignore
import pandas  # type: ignore


def emptyFolder(_path: str) -> None:
    if os.path.exists(_path):
        shutil.rmtree(_path)
    os.mkdir(_path)


def generateBarGraph(_data: dict, _x_label: str, y_label: str, title: str, save_to: str) -> None:
    plt.clf()
    plt.bar(_data.keys(), _data.values())
    plt.xlabel(_x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(save_to)


def generateDelayLegendGraph(
    delay_percentages: dict[str, float],
    total_num_counter: dict[str, int],
    _airport: str,
    _key: str,
    save_to: str,
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
    save_to: str,
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
        delay_percentages[_ky] = (
            df_filtered_with_pushback.shape[0] / df_filtered.shape[0] if df_filtered.shape[0] > 0 else 0
        )
        abs_mean[_ky] = df_filtered["minutes_until_pushback"].mean()
        mean_with_pushback_only[_ky] = df_filtered_with_pushback["minutes_until_pushback"].mean()
    generateDelayLegendGraph(
        delay_percentages,
        total_num_counter,
        _key,
        whichAirport,
        os.path.join(
            save_to,
            "{0}_{1}_frequency.png".format(whichAirport, _key),
        ),
    )
    generateBarGraph(
        abs_mean,
        _key,
        "time (min)",
        "The absolute average delay time every {0} for {1}".format(_key, whichAirport),
        os.path.join(save_to, "{0}_{1}_abs_mean.png".format(whichAirport, _key)),
    )
    generateBarGraph(
        mean_with_pushback_only,
        _key,
        "time (min)",
        "The average delay time every {0} only when there is a delay for {1}".format(_key, whichAirport),
        os.path.join(
            save_to,
            "{0}_{1}_mean_with_pushback_only.png".format(whichAirport, _key),
        ),
    )
