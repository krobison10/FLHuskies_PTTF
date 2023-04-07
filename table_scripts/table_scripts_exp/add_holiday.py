#
# Author: Yudong Lin
#
# extract holiday information
# check if the timestamp given is a holiday
#


import holidays
import pandas as pd  # type: ignore


def add_us_holidays(_df: pd.DataFrame) -> pd.DataFrame:
    us_holidays = holidays.US()
    _df["is_us_holiday"] = _df.apply(lambda x: x.timestamp in us_holidays, axis=1)
    return _df
