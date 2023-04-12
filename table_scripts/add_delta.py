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
    latest_etd: pd.DataFrame = (
        feature_engineering.filter_by_timestamp(data_tables["etd"], now, 30).groupby("gufi").last()
    )
    
    runways: pd.DataFrame = data_tables["runways"].rename(columns={"timestamp": "departure_time"})

    standtimes: pd.DataFrame = data_tables["standtimes"].rename(columns={"timestamp": "departure_time"})

    # How long, on average, passes between the time that departure is estimated to happen
    # and the actual pushback happen
    PDd_1hr = feature_engineering.average_diff_departure_pushback(standtimes,latest_etd, now, 1, "1h_ETDP_delta")
    flights_selected["1h_ETDP"] = pd.Series([PDd_1hr] * len(flights_selected), index=flights_selected.index)

    # How long, on average, passes between the time that actual departure time
    # and the actual pushback happen
    PDd_1hr = feature_engineering.average_diff_departure_pushback(runways,latest_etd, now, 1, "1h_TDP_delta")
    flights_selected["1h_TDP"] = pd.Series([PDd_1hr] * len(flights_selected), index=flights_selected.index)

    return flights_selected