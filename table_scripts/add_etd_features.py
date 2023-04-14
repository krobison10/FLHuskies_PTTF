#
# Author: Daniil Filienko (?)
#
# Add global lamp data
#

import pandas as pd

def add_etd_features(_df: pd.DataFrame,etd:pd.DataFrame, airport:str) -> pd.DataFrame:
    """
    Extracts estimated time of departure features and appends it to the existing dataframe
    :param pd.DataFrame _df: Existing feature set at a timestamp-airport level
    :return pd.Dataframe _df: Master table enlarged with additional features
    """

    etd_features = pd.DataFrame()

    etd["timestamp"] = etd.timestamp.dt.ceil("15min")
    etd["departure_runway_estimated_time"] = pd.to_datetime(
        etd["departure_runway_estimated_time"]
    )
    etd = etd[etd["timestamp"] < etd["departure_runway_estimated_time"]]

    complete_etd = etd.copy()
    for i in range(1, 4 * 25):
        current = etd.copy()
        current["timestamp"] = current["timestamp"] + pd.Timedelta(f"{i * 15}min")
        current = current[
            current["timestamp"] < current["departure_runway_estimated_time"]
        ]
        complete_etd = pd.concat([complete_etd, current])

    complete_etd["time_ahead"] = (
        complete_etd["departure_runway_estimated_time"] - complete_etd["timestamp"]
    ).dt.total_seconds()
    complete_etd = complete_etd.groupby(["gufi", "timestamp"]).first().reset_index()

    for i in [30, 60, 180, 1440]:
        complete_etd[f"estdep_next_{i}min"] = (
            complete_etd["time_ahead"] < i * 60
        ).astype(int)
    complete_etd.sort_values("time_ahead", inplace=True)

    for i in [30, 60, 180, 1440]:
        complete_etd[f"estdep_num_next_{i}min"] = (
            complete_etd["time_ahead"] < i * 60
        ).astype(int)
    complete_etd.sort_values("time_ahead", inplace=True)

    # number of flights departing from the airport 
    etd_aggregation = (
        complete_etd.groupby("timestamp")
        .agg(
            {
                "gufi": "count",
                "estdep_next_30min": "sum",
                "estdep_next_1hr": "sum",
                "estdep_next_3hrs": "sum",
                "estdep_next_1day": "sum",
            }
        )
        .reset_index()
    )

    etd_aggregation.columns = [
        "feat_5_" + c if c != "timestamp" else c for c in etd_aggregation.columns
    ]
    etd_features = pd.concat([etd_features, etd_aggregation])

    _df = _df.merge(
        etd_features, how="left", on=["timestamp"]
    )

    return _df
