#
# Author: Yudong Lin
#
# extract date based features from the timestamp
#

import holidays
import pandas as pd  # type: ignore


def add_date_features(_df: pd.DataFrame) -> pd.DataFrame:
    _df["year"] = _df.apply(lambda x: x.timestamp.year, axis=1)
    _df["month"] = _df.apply(lambda x: x.timestamp.month, axis=1)
    _df["day"] = _df.apply(lambda x: x.timestamp.day, axis=1)
    _df["hour"] = _df.apply(lambda x: x.timestamp.hour, axis=1)
    _df["minute"] = _df.apply(lambda x: x.timestamp.minute, axis=1)
    _df["weekday"] = _df.apply(lambda x: x.timestamp.weekday(), axis=1)

    # check if the timestamp given is a holiday
    us_holidays = holidays.US()
    _df["is_us_holiday"] = _df.apply(lambda x: x.timestamp in us_holidays, axis=1)

    return _df
