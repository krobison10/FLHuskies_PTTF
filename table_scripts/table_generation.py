#
# Authors:
# - Kyler Robison
# - Yudong Lin
# - Daniil Filienko
# generate the full table for a specific airport
#

import feature_engineering
import pandas as pd
from add_averages import add_averages
from add_config import add_config
from add_date import add_date_features
from add_etd import add_etd
from add_etd_features import add_etd_features
from add_lamp import add_lamp
from add_traffic import add_traffic
from extract_gufi_features import extract_and_add_gufi_features
from feature_tables import get_feature_tables
from utils import get_csv_path


def _process_timestamp(filtered_table: pd.DataFrame, data_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    # subset table to only contain flights for the current timestamp
    filtered_table = filtered_table.reset_index(drop=True)

    # get current time
    now: pd.Timestamp = filtered_table.timestamp.iloc[0]

    # filters the data tables to only include data from past 30 hours, this call can be omitted in a submission script
    data_tables = filter_tables(now, data_tables)

    # get the latest ETD for each flight
    latest_etd: pd.DataFrame = data_tables["etd"].groupby("gufi").last()

    # add features
    filtered_table = add_etd(filtered_table, latest_etd)
    filtered_table = add_averages(now, filtered_table, latest_etd, data_tables)
    filtered_table = add_traffic(now, filtered_table, latest_etd, data_tables)
    filtered_table = add_config(filtered_table, data_tables)
    filtered_table = add_lamp(now, filtered_table, data_tables)

    return filtered_table


def filter_tables(now: pd.Timestamp, data_tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    new_dict: dict[str, pd.DataFrame] = {}

    for key in data_tables:
        if not key.endswith("mfs"):
            new_dict[key] = feature_engineering.filter_by_timestamp(data_tables[key], now, 30)

    new_dict["mfs"] = filter_mfs(data_tables["mfs"], new_dict["standtimes"])
    new_dict["private_mfs"] = filter_mfs(data_tables["private_mfs"], new_dict["private_standtimes"])

    return new_dict


def filter_mfs(mfs: pd.DataFrame, standtimes: pd.DataFrame) -> pd.DataFrame:
    return mfs.loc[mfs["gufi"].isin(standtimes["gufi"])]


def generate_table(_airport: str, data_dir: str, max_rows: int = -1) -> pd.DataFrame:
    from pandarallel import pandarallel  # type: ignore

    pandarallel.initialize(verbose=1, progress_bar=True)

    # read train labels for given airport
    _df: pd.DataFrame = pd.read_csv(
        get_csv_path(data_dir, "train_labels_phase2", f"phase2_train_labels_{_airport}.csv"),
        parse_dates=["timestamp"],
    )

    # if you want to select only a certain amount of row
    if max_rows > 0:
        _df = _df[:max_rows]

    # define list of data tables to load and use for each airport
    feature_tables: dict[str, pd.DataFrame] = get_feature_tables(data_dir, _airport)

    # process all prediction times in parallel
    _df = _df.groupby("timestamp").parallel_apply(lambda x: _process_timestamp(x, feature_tables))

    # Add runway information
    # _df = _df.merge(feature_tables["runways"][["gufi", "departure_runway_actual"]], how="left", on="gufi")

    # extract and add mfs information
    _df = extract_and_add_gufi_features(_df)

    # extract holiday features
    _df = add_date_features(_df)

    # Add additional etd features
    _df = add_etd_features(_df, feature_tables["etd"])

    # Add mfs information
    _df = _df.merge(feature_tables["mfs"], how="left", on="gufi")

    return _df
