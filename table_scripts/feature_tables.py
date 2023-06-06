#
# Author: Yudong Lin
#
# read all feature tables

from glob import glob
import pandas as pd
from utils import get_csv_path


def get_feature_tables(data_dir: str, _airport: str) -> dict[str, pd.DataFrame]:
    return {
        "etd": pd.read_csv(
            get_csv_path(data_dir, "public", _airport, f"{_airport}_etd.csv"),
            parse_dates=["departure_runway_estimated_time", "timestamp"],
        ).sort_values("timestamp"),
        "config": pd.read_csv(
            get_csv_path(data_dir, "public", _airport, f"{_airport}_config.csv"),
            parse_dates=["timestamp", "start_time"],
            dtype={"departure_runways": "category", "arrival_runways": "category"},
        ).sort_values("timestamp", ascending=False),
        "first_position": pd.read_csv(
            get_csv_path(data_dir, "public", _airport, f"{_airport}_first_position.csv"), parse_dates=["timestamp"]
        ),
        "lamp": pd.read_csv(
            get_csv_path(data_dir, "public", _airport, f"{_airport}_lamp.csv"),
            parse_dates=["timestamp", "forecast_timestamp"],
            dtype={
                "cloud_ceiling": "int8",
                "visibility": "int8",
                "cloud": "category",
                "lightning_prob": "category",
                "precip": bool,
            },
        )
        .set_index("timestamp", drop=False)
        .sort_index(),
        "runways": pd.read_csv(
            get_csv_path(data_dir, "public", _airport, f"{_airport}_runways.csv"),
            dtype={"departure_runway_actual": "category", "arrival_runway_actual": "category"},
            parse_dates=["timestamp", "departure_runway_actual_time", "arrival_runway_actual_time"],
        ),
        "standtimes": pd.read_csv(
            get_csv_path(data_dir, "public", _airport, f"{_airport}_standtimes.csv"),
            parse_dates=["timestamp", "departure_stand_actual_time"],
        ).merge(
            pd.concat(
                [
                    pd.read_csv(
                        get_csv_path(airline_standtimes),
                        parse_dates=["timestamp", "arrival_stand_actual_time", "departure_stand_actual_time"],
                    )
                    for airline_standtimes in glob(get_csv_path(data_dir, "private", _airport, f"*_standtimes.csv"))
                ]
            ),
            how="left",
            on=["gufi", "timestamp", "departure_stand_actual_time"],
        ),
        "mfs": pd.concat(
            [
                pd.read_csv(
                    get_csv_path(airline_standtimes),
                    dtype={
                        "aircraft_engine_class": "category",
                        "aircraft_type": "category",
                        "major_carrier": "category",
                        "flight_type": "category",
                        "isdeparture": bool,
                    },
                )
                for airline_standtimes in glob(get_csv_path(data_dir, "private", _airport, f"*_mfs.csv"))
            ]
        ),
    }
