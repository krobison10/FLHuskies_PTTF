#
# Author: Yudong Lin
#
# extract date based features from the timestamp
#

import polars as pl


def add_date_features(_df: pl.DataFrame) -> pl.DataFrame:
    _df = _df.with_columns(
        [
            (pl.col("timestamp").dt.year()).alias("year"),
            # (pl.col("timestamp").dt.quarter()).alias("quarter"),
            (pl.col("timestamp").dt.month()).alias("month"),
            (pl.col("timestamp").dt.day()).alias("day"),
            (pl.col("timestamp").dt.weekday()).alias("weekday"),
            (pl.col("timestamp").dt.hour()).alias("hour"),
            (pl.col("timestamp").dt.minute()).alias("minute"),
        ]
    )

    # check if the timestamp given is a holiday
    # us_holidays = holidays.US()
    # _df["is_us_holiday"] = _df.apply(lambda x: x.timestamp in us_holidays, axis=1)

    return _df
