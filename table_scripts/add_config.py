#
# Author: Jeff Maloney, Kyler Robison, Daniil Filienko
#
# calculate and add airport runway configurations to the data frame
#

import feature_engineering
import pandas as pd  # type: ignore
import numpy as np

# find and add current runway configuration
def add_config(flights_selected: pd.DataFrame, data_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    # filter features to 30 hours before prediction time to prediction time and save as a copy
    config: pd.DataFrame = data_tables["config"]

    # most recent row of airport configuration
    if config.shape[0] > 0:
        departure_runways = config["departure_runways"].iloc[0]
        arrival_runways = config["arrival_runways"].iloc[0]
    else:
        departure_runways = "UNK"
        arrival_runways = "UNK"

    # add new column for which departure runways are in use at this timestamp
    flights_selected["departure_runways"] = pd.Series(
        [departure_runways] * len(flights_selected), index=flights_selected.index
    )
    flights_selected["arrival_runways"] = pd.Series(
        [arrival_runways] * len(flights_selected), index=flights_selected.index
    )

    flights_selected = add_ratios(config, flights_selected)

    return flights_selected



def add_ratios(config: pd.DataFrame, flights_selected: pd.DataFrame):
    # Find the maximum count across all columns
    max_count_dep = (flights_selected["departure_runways"].str.count(',').max()) + 1

    # Find the maximum count across all columns
    max_count_arr = (flights_selected["arrival_runways"].str.count(',').max()) + 1

    # Get the ratio of how much of the departure runways are used
    flights_selected['dep_ratio'] = ((flights_selected["departure_runways"].astype(str).str.count(',') + 1) / max_count_dep)

    # Replace NaN values in dep_ratio column with closest previous available value
    flights_selected['dep_ratio'].fillna(method='ffill', inplace=True)

    # Get the ratio of how much of the arrival runways are used
    flights_selected['arr_ratio'] = ((flights_selected["arrival_runways"].astype(str).str.count(',') + 1) / max_count_arr)

    # Replace NaN values in arr_ratio column with closest previous available value
    flights_selected['arr_ratio'].fillna(method='ffill', inplace=True)

    return flights_selected
