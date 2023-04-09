#
# Author: Daniil Filienko (?)
#
# Add global lamp data
#

import pandas as pd

# add global lamp forecast weather information with 6 hour moving window of
# av, stdt, max, and min, based on the historic trends
def add_global_lamp(_df: pd.DataFrame,raw_data:pd.DataFrame, airport:str) -> pd.DataFrame:
    """
    Extracts features of weather forecasts for each airport and appends it to the
    existing master table
    :param pd.Dataframe _df: Existing feature set at a timestamp-airport level
    :return pd.Dataframe _df: Master table enlarged with additional features
    """

    weather = pd.DataFrame()
    
    current = raw_data.copy()

    current["forecast_timestamp"] = pd.to_datetime(current["forecast_timestamp"])

    current["timestamp"] = pd.to_datetime(current["timestamp"])

    current["lightning_prob"] = current["lightning_prob"].map(
        {"L": 0, "M": 1, "N": 2, "H": 3}
    )
    current["cloud"] = (
        current["cloud"]
        .map({"OV": 4, "BK": 3, "CL": 0, "FW": 1, "SC": 2})
        .fillna(3)
    )
    current["precip"] = current["precip"].astype(float)
    current["time_ahead_prediction"] = (
        current["forecast_timestamp"] - current["timestamp"]
    ).dt.total_seconds() / 3600
    current.sort_values(["timestamp", "time_ahead_prediction"], inplace=True)

    past_temperatures = (
        current.groupby("timestamp")
        .first()
        .drop(columns=["forecast_timestamp", "time_ahead_prediction"])
    )
    past_temperatures = (
        past_temperatures.rolling("6h").agg({"mean", "min", "max"}).reset_index()
    )
    past_temperatures.columns = [
        "feats_lamp_" + c[0] + "_" + c[1] + "_last6h"
        if c[0] != "timestamp"
        else "timestamp"
        for c in past_temperatures.columns
    ]
    past_temperatures = (
        past_temperatures.set_index("timestamp")
        .resample("15min")
        .ffill()
        .reset_index()
    )

    current_feats = past_temperatures.copy()

    for p in range(1, 24):
        next_temp = (
            current[
                (current.time_ahead_prediction <= p)
                & (current.time_ahead_prediction > p - 1)
            ]
            .drop(columns=["forecast_timestamp", "time_ahead_prediction"])
            .groupby("timestamp")
            .mean()
            .reset_index()
        )
        next_temp.columns = [
            "feat_lamp_" + c + "_next_" + str(p) if c != "timestamp" else "timestamp"
            for c in next_temp.columns
        ]
        next_temp = (
            next_temp.set_index("timestamp").resample("15min").ffill().reset_index()
        )
        current_feats = current_feats.merge(next_temp, how="left", on="timestamp")

    current_feats["airport"] = airport

    weather = pd.concat([weather, current_feats])

    _df = _df.merge(weather, how="left", on=["airport", "timestamp"])

    # Add global weather features
    weather_feats = [c for c in weather.columns if "feat_lamp" in c]
    for feat in weather_feats:
        _df[feat + "_global_min"] = _df["timestamp"].map(
            weather.groupby("timestamp")[feat].min()
        )
        _df[feat + "_global_mean"] = _df["timestamp"].map(
            weather.groupby("timestamp")[feat].mean()
        )
        _df[feat + "_global_max"] = _df["timestamp"].map(
            weather.groupby("timestamp")[feat].max()
        )
        _df[feat + "_global_std"] = _df["timestamp"].map(
            weather.groupby("timestamp")[feat].std()
        )

    return _df