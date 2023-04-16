#
# Author: Daniil Filienko
#
# calculate and add flight duration information to the general data frame
#

import pandas as pd  # type: ignore

def add_estimated_flight_time(flights_selected: pd.DataFrame, data_tables: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    The estimated time it takes for the flight to happen
    Available for the predicted flight
    """
    latest_tfm: pd.DataFrame = data_tables["tfm"].groupby("gufi").last()

    latest_etd: pd.DataFrame = data_tables["etd"].groupby("gufi").last()

    # Estimated flight duration for all of the flights within last 30 hours
    flights_selected["flight_time"] = (
        latest_tfm["arrival_runway_estimated_time"] - latest_etd["estimated_runway_departure_time"]
    ).dt.total_seconds() / 60

    flights_selected = flights_selected["flight_time"].fillna(0)

    return flights_selected
