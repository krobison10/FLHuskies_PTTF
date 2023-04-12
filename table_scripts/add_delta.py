#
# Author: Daniil Filienko
#
# calculate the delta between ETD and Actual Pushback Time 
# to see how the late/early an average nearby flight may be
#

import feature_engineering
import pandas as pd  # type: ignore


# calculate the 1 hr. av difference
def add_delta(
    now: pd.Timestamp, flights_selected: pd.DataFrame, data_tables: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    latest_etd: pd.DataFrame = data_tables["etd"].groupby("gufi").last()
    
    standtimes: pd.DataFrame = data_tables["standtimes"]

    # How long, on average, passes between the time that departure is estimated to happen
    # and the actual pushback happen
    PDd_1hr = feature_engineering.average_departure_delay(latest_etd,standtimes, now, 1, "departure_stand_actual_time")
    flights_selected["1h_ETDP"] = pd.Series([PDd_1hr] * len(flights_selected), index=flights_selected.index)

    return flights_selected