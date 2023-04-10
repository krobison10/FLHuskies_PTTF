#
# Author: Jeff Maloney, Kyler Robison, Daniil Filienko
#
# calculate and add airport runway configurations to the data frame
#

import feature_engineering
import pandas as pd  # type: ignore
import numpy as np

# find and add current runway configuration
def add_config(now: pd.Timestamp, flights_selected: pd.DataFrame, data_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    # filter features to 30 hours before prediction time to prediction time and save as a copy
    config: pd.DataFrame = feature_engineering.filter_by_timestamp(data_tables["config"], now, 30)

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

    add_ratios(config, flights_selected)

    return flights_selected



def add_ratios(config: pd.DataFrame, flights_selected: pd.DataFrame):
    # Find the maximum count across all columns
    max_count_dep = (config["departure_runways"].str.count(',').max()) + 1

    # Find the maximum count across all columns
    max_count_arr = (config["arrival_runways"].str.count(',').max()) + 1

    # Find the mean dep runways count across all columns
    mean_count_dep = config['departure_runways'].str.count(',').mean()

    # Find the mean arr runways  count across all columns
    mean_count_arr = config['arrival_runways'].str.count(',').mean()

    # Get the ration of how much of the departure runways are used
    flights_selected['dep_ratio'] = ((config["departure_runways"].str.count(',') + 1) / max_count_dep)

    # Replacing non-existent values with the average
    flights_selected['dep_ratio'] = np.where(flights_selected['dep_ratio'].isna(), mean_count_dep / max_count_dep,
                                             flights_selected['dep_ratio'])

    # Get the ration of how much of the arrival runways are used
    flights_selected['arr_ratio'] = ((config["arrival_runways"].str.count(',') + 1) / max_count_arr)

    # Replacing non-existent values with the average
    flights_selected['arr_ratio'] = np.where(flights_selected['arr_ratio'].isna(), mean_count_arr / max_count_arr,
                                             flights_selected['arr_ratio'])
