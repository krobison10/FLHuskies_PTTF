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
  
    # Find the maximum count across all columns
    max_count_dep = (config["departure_runways"].str.count(',').max()) + 1

    # Find the maximum count across all columns
    max_count_arr = (config["arrival_runways"].str.count(',').max()) + 1

    # Find the mean dep runways count across all columns
    mean_count_dep = config['departure_runways'].str.count(',').mean()

    # Find the mean arr runways  count across all columns
    mean_count_arr = config['arrival_runways'].str.count(',').mean()

    # most recent row of airport configuration
    # config_string = config.head(1)['departure_runways'].to_string(index=False)
    if config.shape[0] > 0:
        config_string = config["departure_runways"].iloc[0]
    else:
        config_string = "UNK"

    # add new column for which departure runways are in use at this timestamp
    flights_selected["departure_runways"] = pd.Series(
        [config_string] * len(flights_selected), index=flights_selected.index
    )

    # Get the ration of how much of the departure runways are used
    flights_selected['dep_ratio'] = ((config["departure_runways"].str.count(',') + 1) / max_count_dep)
    
    # Replacing non existent values with the average
    flights_selected['dep_ratio'] = np.where(flights_selected['dep_ratio'].isna(), mean_count_dep/max_count_dep, flights_selected['dep_ratio'])

    # Get the ration of how much of the arrival runways are used
    flights_selected['arr_ratio'] = ((config["arrival_runways"].str.count(',') + 1) / max_count_arr)

    # Replacing non existent values with the average
    flights_selected['arr_ratio'] = np.where(flights_selected['arr_ratio'].isna(), mean_count_arr/max_count_arr, flights_selected['arr_ratio'])

    return flights_selected