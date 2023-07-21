#
# Authors:
# - Kyler Robison
# - Yudong Lin
#
# A set of helper functions
#

import os

import pandas as pd


# get a valid path for a csv file
# try to return the path for uncompressed csv file first
# if the uncompressed csv does not exists, then return the path for compressed csv file
def get_csv_path(*argv: str) -> str:
    etd_csv_path: str = os.path.join(*argv)
    if not os.path.exists(etd_csv_path):
        etd_csv_path += ".bz2"
    return etd_csv_path


# specify an airport to split for only one, otherwise a split for all airports will be executed
def train_test_split(table: pd.DataFrame, ROOT: str, dirs: dict[str, str], airport: str, _k: str) -> None:
    val_data: pd.DataFrame = pd.read_csv(os.path.join(ROOT, "_data", "submission_format.csv"))

    # If there is a specific airport then we are only interested in those rows
    if airport != "ALL":
        val_data = val_data[val_data.airport == airport]

    _gufi = val_data.gufi.unique()
    test_data: pd.DataFrame = table[table.gufi.isin(_gufi)]
    train_data: pd.DataFrame = table[~table.gufi.isin(_gufi)]

    # replace these paths with any desired ones if necessary
    train_data.sort_values(["gufi", "timestamp"]).to_csv(
        os.path.join(dirs["train_tables"], f"{_k}_train.csv"), index=False
    )
    test_data.sort_values(["gufi", "timestamp"]).to_csv(
        os.path.join(dirs["validation_tables"], f"{_k}_validation.csv"), index=False
    )

encoded_columns = [
    "cloud",
    "lightning_prob",
    "precip",
    #"gufi_flight_major_carrier",
    "gufi_flight_destination_airport",
    "aircraft_engine_class",
    "aircraft_type",
    "major_carrier",
    "flight_type",
    #"airport"
]

features = [
    #"airport",
    #"gufi_flight_major_carrier",
    "deps_3hr",
    "deps_30hr",
    #"arrs_3hr",
    #"arrs_30hr",
    "deps_taxiing",
    #"arrs_taxiing",
    "exp_deps_15min",
    "exp_deps_30min",
    "standtime_30hr",
    "dep_taxi_30hr",
    #"arr_taxi_30hr",
    "minute",
    "gufi_flight_destination_airport",
    "month",
    "day",
    "hour",
    "year",
    "weekday",
    "minutes_until_etd",
    "aircraft_engine_class",
    "aircraft_type",
    "major_carrier",
    "flight_type",
    "temperature",
    "wind_direction",
    "wind_speed",
    "wind_gust",
    "cloud_ceiling",
    "visibility",
    "cloud",
    "lightning_prob",
    "precip",
]
