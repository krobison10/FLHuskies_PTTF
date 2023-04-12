#
# Author: Kyler Robison
#
# calculate and add etd information to the data frame
#

import feature_engineering
import pandas as pd  # type: ignore


# calculate etd
def add_etd(flights_selected: pd.DataFrame, data_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    final_table = flights_selected

    etd = data_tables["etd"]

    # get the latest ETD for each flight
    latest_etd: pd.DataFrame = etd.groupby("gufi").last()

    # get a series containing latest ETDs for each flight, in the same order they appear in flights
    departure_runway_estimated_time: pd.Series = flights_selected.merge(
        latest_etd.departure_runway_estimated_time, how="left", on="gufi"
    ).departure_runway_estimated_time

    # add new column to flights_selected that represents minutes until pushback
    final_table["minutes_until_etd"] = (
        (departure_runway_estimated_time - flights_selected.timestamp).dt.total_seconds() / 60
    ).astype(int)

    return final_table
