#
# Authors:
# - Kyler Robison
# - Yudong Lin
#
# A set of helper functions
#

import os

import polars as pl


# get a valid path for a csv file
# try to return the path for uncompressed csv file first
# if the uncompressed csv does not exists, then return the path for compressed csv file
def get_csv_path(*argv: str) -> str:
    etd_csv_path: str = os.path.join(*argv)
    if not os.path.exists(etd_csv_path):
        etd_csv_path += ".bz2"
    return etd_csv_path


# specify an airport to split for only one, otherwise a split for all airports will be executed
def train_test_split(table: pl.DataFrame, ROOT: str, airport: str | None = None) -> None:
    val_data: pl.DataFrame = pl.read_csv(os.path.join(ROOT, "data", "submission_format.csv"))

    # If there is a specific airport then we are only interested in those rows
    ext: str = "ALL"
    if airport is not None:
        ext = f"{airport}"
        val_data = val_data.filter(pl.col("airport") == airport)

    _gufi: pl.Series = val_data.select(pl.col("gufi")).unique().to_series()
    test_data: pl.DataFrame = table.filter(pl.col("gufi").is_in(_gufi))
    train_data: pl.DataFrame = table.filter(~pl.col("gufi").is_in(_gufi))

    # replace these paths with any desired ones if necessary
    train_data.sort("gufi", "timestamp").write_csv(os.path.join(ROOT, "train_tables", f"{ext}_train.csv"))
    test_data.sort("gufi", "timestamp").write_csv(os.path.join(ROOT, "validation_tables", f"{ext}_validation.csv"))


def add_difference_in_minutes(df: pl.DataFrame, col1: str, col2: str, alias_col_name: str) -> pl.DataFrame:
    df = df.with_columns((pl.col(col1) - pl.col(col2)).alias(alias_col_name))
    return df.with_columns(pl.col(alias_col_name).apply(lambda x: x.total_seconds() // 60))


def get_mean_of_col(df: pl.DataFrame, _col: str, round_to: int = 2) -> float:
    avg_delay: int | float | None = df.select(pl.col(_col)).to_series().mean()
    return round(avg_delay if avg_delay is not None else 0, round_to)


def get_average_difference_in_minutes(df1: pl.DataFrame, df2: pl.DataFrame, col1: str, col2: str) -> float:
    return get_mean_of_col(
        add_difference_in_minutes(df1.join(df2, on="gufi"), col1, col2, "difference_temp_data"), "difference_temp_data"
    )
