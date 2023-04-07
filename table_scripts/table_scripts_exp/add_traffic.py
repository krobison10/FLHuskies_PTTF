#
# Author: Kyler Robison
#
# calculate and add traffic information to data table
#

import feature_engineering
import pandas as pd  # type: ignore


# calculate various traffic measures for airport
def add_traffic(now: pd.Timestamp, flights_selected: pd.DataFrame, data_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return flights_selected
