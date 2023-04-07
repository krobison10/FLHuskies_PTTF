#
# Author: Jeff Maloney, Kyler Robison
#
# calculate and add airport runway configurations to the data frame
#

import feature_engineering
import pandas as pd  # type: ignore


# find and add current runway configuration
def add_config(now: pd.Timestamp, flights_selected: pd.DataFrame, data_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    # filter features to 30 hours before prediction time to prediction time and save as a copy
    config: pd.DataFrame = feature_engineering.filter_by_timestamp(data_tables["config"], now, 30)

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

    return flights_selected
