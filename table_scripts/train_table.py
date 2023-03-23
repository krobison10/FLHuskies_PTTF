#
# Author: Kyler Robison
#
# This script builds a table of training data for a single airport that is hard coded.
# It can easily be changed.
#
# To run on compressed data with format specified in README.md, supply a command line
# argument "compressed" (untested).
#

import sys
import multiprocessing
import pandas as pd
import feature_engineering

from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import OrdinalEncoder
from functools import partial
from pathlib import Path
from tqdm import tqdm

ext = ""
if sys.argv[1] == "compressed":
    ext = ".bz2"

DATA_DIR = Path("../_data")
airport = "KSEA"

table = pd.read_csv(DATA_DIR / airport / f"train_labels_{airport}.csv{ext}", parse_dates=["timestamp"])


def process_timestamp(
        now: pd.Timestamp,
        flights: pd.DataFrame,
        data_tables: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    # subset table to only contain flights for the current timestamp
    time_filtered_table: pd.DataFrame = flights.loc[flights.timestamp == now].reset_index(drop=True)

    final_table = time_filtered_table.copy()

    # filter features to 30 hours before prediction time to prediction time and save as a copy
    etd: pd.DataFrame = feature_engineering.filter_by_timestamp(data_tables['etd'], now, 30).copy()
    origin: pd.DataFrame = feature_engineering.filter_by_timestamp(data_tables['first_position'], now, 30).copy()
    standtimes: pd.DataFrame = feature_engineering.filter_by_timestamp(data_tables['standtimes'], now, 30).copy()
    runways: pd.DataFrame = data_tables['runways']

    # rename origin timestamp to origin_time as to not get confused in future joins,
    # because timestamp is the important feature
    origin = origin.rename(columns={'timestamp': 'origin_time'})

    # ----- Minutes Until ETD -----
    # get the latest ETD for each flight
    latest_etd: pd.DataFrame = etd.groupby("gufi").last()

    # get a series containing latest ETDs for each flight, in the same order they appear in flights
    departure_runway_estimated_time: pd.Series = time_filtered_table.merge(
        latest_etd.departure_runway_estimated_time, how="left", on="gufi"
    ).departure_runway_estimated_time

    # add new column to time_filtered_table that represents minutes until pushback
    final_table["minutes_until_etd"] = (
        ((departure_runway_estimated_time - time_filtered_table.timestamp).dt.total_seconds() / 60).astype(int)
    )

    # ----- Minutes Since Origin (WIP) -----
    # get a series containing origin time for each flight, in the same order they appear in flights
    # origin_time: pd.Series = time_filtered_table.merge(
    #     origin, how="left", on="gufi"
    # ).origin_time

    # add new column to time_filtered_table that represents minutes since origin
    # time_filtered_table["minutes_since_origin"] = (
    #     ((time_filtered_table.timestamp - origin_time).dt.total_seconds() / 60).astype(int)
    # )

    # ----- 3hr Average Delay -----
    delay_3hr = feature_engineering.average_departure_delay(latest_etd, runways, now, 3)
    final_table['delay_3hr'] = pd.Series([delay_3hr] * len(time_filtered_table),
                                         index=time_filtered_table.index)

    # ----- 30hr Average Delay -----
    delay_30hr = feature_engineering.average_departure_delay(latest_etd, runways, now, 30)
    final_table['delay_30hr'] = pd.Series([delay_30hr] * len(time_filtered_table),
                                          index=time_filtered_table.index)

    # ----- 3hr Average Time at Stand -----
    standtime_3hr = feature_engineering.average_stand_time(origin, standtimes, now, 3)
    final_table['standtime_3hr'] = pd.Series([standtime_3hr] * len(time_filtered_table),
                                             index=time_filtered_table.index)

    # ----- 30hr Average Time at Stand -----
    standtime_30hr = feature_engineering.average_stand_time(origin, standtimes, now, 30)
    final_table['standtime_30hr'] = pd.Series([standtime_30hr] * len(time_filtered_table),
                                              index=time_filtered_table.index)

    return final_table


# define list of data tables to load and use for each airport
feature_tables: dict[str, pd.DataFrame] = {
    'etd': pd.read_csv(
        DATA_DIR / airport / f"features/{airport}_etd.csv{ext}",
        parse_dates=["departure_runway_estimated_time", "timestamp"]
    ).sort_values("timestamp"),
    'runways': pd.read_csv(
        DATA_DIR / airport / f"features/{airport}_runways.csv{ext}",
        parse_dates=["departure_runway_actual_time", "timestamp"]
    ),
    'first_position': pd.read_csv(
        DATA_DIR / airport / f"features/{airport}_first_position.csv{ext}",
        parse_dates=["timestamp"]
    ),
    'standtimes': pd.read_csv(
        DATA_DIR / airport / f"features/{airport}_standtimes.csv{ext}",
        parse_dates=["timestamp", "departure_stand_actual_time"]
    )
}

# Add encoded column for runway
table['departure_runway'] = table.merge(feature_tables['runways'], how='left', on='gufi').departure_runway_actual
encoder = OrdinalEncoder()
encoded_runways = encoder.fit_transform(table[['departure_runway']])
table['departure_runway'] = encoded_runways

# process all prediction times in parallel
with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
    fn = partial(process_timestamp, flights=table, data_tables=feature_tables)
    timestamp_tables: list[pd.DataFrame] = list(
        tqdm(
            executor.map(
                fn,
                pd.to_datetime(table.timestamp.unique())
            ), total=len(table.timestamp.unique())
        )
    )

# concatenate individual prediction times to a single dataframe
table: pd.DataFrame = pd.concat(timestamp_tables, ignore_index=True)

# move train label column to the end
cols = table.columns.tolist()
cols.remove('minutes_until_pushback')
cols.append('minutes_until_pushback')
table = table[cols]

# save with name "main.csv"
table.to_csv(Path("../train_tables/main.csv"), index=False)
