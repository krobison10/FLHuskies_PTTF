#
# Author: Yudong Lin
#
# obtain the latest forecasts information and add it to the data frame
#

from typing import Optional

import pandas as pd  # type: ignore


def _look_for_forecasts(_lamp: pd.DataFrame, _look_for_timestamp: pd.Timestamp, _now: pd.Timestamp) -> Optional[pd.DataFrame]:
    # select all rows contain this forecast_timestamp
    forecasts: pd.DataFrame = _lamp.loc[
        (_lamp.forecast_timestamp == _look_for_timestamp) & (_now - pd.Timedelta(hours=30) <= _lamp.index) & (_lamp.index <= _now)
    ]
    # get the latest forecast
    return forecasts.iloc[forecasts.index.get_indexer([_now], method="nearest")] if forecasts.shape[0] > 0 else None


# add lamp forecast weather information
def add_lamp(now: pd.Timestamp, flights_selected: pd.DataFrame, data_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    _lamp: pd.DataFrame = data_tables["lamp"]
    # the latest forecast
    latest_forecast: Optional[pd.DataFrame] = None
    # counter to monitoring hours going forward
    hour_f: int = 0
    # when no valid forecast is found
    while latest_forecast is None:
        # round time to the nearest hour
        forecast_timestamp_look_up = (now - pd.Timedelta(hours=hour_f)).round("H")
        # get the latest forecast
        latest_forecast = _look_for_forecasts(_lamp, forecast_timestamp_look_up, now)
        # if a valid latest forecast is found
        if latest_forecast is not None:
            # then update value
            for key in ("temperature", "wind_direction", "wind_speed", "wind_gust", "cloud_ceiling", "visibility", "cloud", "lightning_prob", "precip"):
                flights_selected[key] = latest_forecast[key].values[0]
            # and break the loop
            break
        # if no forecast within 30 hours can be found
        elif hour_f > 30:
            for key in ("temperature", "wind_direction", "wind_speed", "wind_gust", "cloud_ceiling", "visibility"):
                flights_selected[key] = 0
            for key in ("cloud", "lightning_prob", "precip"):
                flights_selected[key] = "UNK"
            break
        hour_f += 1

    return flights_selected
