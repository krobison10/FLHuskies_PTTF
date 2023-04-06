#
# Author: Yudong Lin
#
# extract holiday information
# check if the timestamp given is a holiday
#


import pandas as pd  # type: ignore
from pandas.tseries.holiday import USFederalHolidayCalendar  # type: ignore


def add_us_holidays(_df: pd.DataFrame) -> pd.DataFrame:
    _cal: USFederalHolidayCalendar = USFederalHolidayCalendar()
    holidays = _cal.holidays()
    _df["is_us_holiday"] = _df.timestamp.isin(holidays)
    return _df
