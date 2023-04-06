#
# Author: Yudong Lin
#
# A simple script that Yudong use to evaluate all features
#

import mytools

features: tuple[str, ...] = (
    "temperature",
    "wind_direction",
    "wind_speed",
    "wind_gust",
    "delay_3hr",
    "delay_30hr",
    "standtime_3hr",
    "standtime_30hr",
    "cloud_ceiling",
    "visibility",
    "month",
    "day",
    "hour",
    "weekday",
    "minutes_until_etd",
    "minutes_until_pushback",
)

_data = mytools.get_train_tables()

mytools.evaluate_numerical_features(_data, features)
