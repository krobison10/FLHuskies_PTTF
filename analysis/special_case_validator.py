import os

import numpy
import pandas as pd  # type: ignore

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

DATA_DIR: str = os.path.join(os.path.dirname(__file__), "..", "train_tables")

for airport in airports:
    print("Checking", airport)
    table: pd.DataFrame = pd.read_csv(
        os.path.join(DATA_DIR, f"main_{airport}_prescreened.csv"),
        parse_dates=["timestamp"],
        dtype={"precip": str, "isdeparture": str},
        low_memory=False,
    )

    table = table.drop_duplicates(subset=["gufi"])

    when_is_departure_is_false: pd.DataFrame = table.loc[table.isdeparture == "False"]
    assert table.loc[table.isdeparture == "True"].shape[0] > 0
    if when_is_departure_is_false.shape[0] > 0:
        print(when_is_departure_is_false)

    """
    when_precip_is_True: pd.DataFrame = table.loc[table.precip == "True"]
    if when_precip_is_True.shape[0] > 0:
        print(when_precip_is_True)
    """

    gufi_TFM_TFDM: pd.DataFrame = table.loc[table.gufi.str.endswith("TFM_TFDM") == True]
    gufi_TFM: pd.DataFrame = table.loc[table.gufi.str.endswith("TFM") == True]
    gufi_TMA: pd.DataFrame = table.loc[table.gufi.str.endswith("TMA") == True]
    gufi_OTHER: pd.DataFrame = table.loc[
        (table.gufi.str.endswith("TFM") == False)
        & (table.gufi.str.endswith("TFM_TFDM") == False)
        & (table.gufi.str.endswith("TMA") == False)
    ]
    print(
        f"TMA Info (min, avg, max) for {gufi_TMA.shape[0]} labels:",
        numpy.min(gufi_TMA["minutes_until_pushback"]),
        numpy.average(gufi_TMA["minutes_until_pushback"]),
        numpy.max(gufi_TMA["minutes_until_pushback"]),
    )
    print(
        f"TFM_TFDM Info (min, avg, max) for {gufi_TFM_TFDM.shape[0]} labels:",
        numpy.min(gufi_TFM_TFDM["minutes_until_pushback"]),
        numpy.average(gufi_TFM_TFDM["minutes_until_pushback"]),
        numpy.max(gufi_TFM_TFDM["minutes_until_pushback"]),
    )
    print(
        f"TFM Info (min, avg, max) for {gufi_TFM.shape[0]} labels:",
        numpy.min(gufi_TFM["minutes_until_pushback"]),
        numpy.average(gufi_TFM["minutes_until_pushback"]),
        numpy.max(gufi_TFM["minutes_until_pushback"]),
    )
    if gufi_OTHER.shape[0] > 0:
        print(
            f"Other Info (min, avg, max) for {gufi_OTHER.shape[0]} labels:",
            numpy.min(gufi_OTHER["minutes_until_pushback"]),
            numpy.average(gufi_OTHER["minutes_until_pushback"]),
            numpy.max(gufi_OTHER["minutes_until_pushback"]),
        )
    else:
        print("No flight with other label.")
