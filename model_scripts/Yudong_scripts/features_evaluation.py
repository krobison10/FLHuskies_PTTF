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
    "minute",
    "weekday",
    "minutes_until_etd",
    "minutes_until_pushback",
)

_data = mytools.get_train_tables().drop_duplicates(subset=["gufi"])

_data["month"] = _data.apply(lambda x: x.timestamp.month, axis=1)
_data["day"] = _data.apply(lambda x: x.timestamp.day, axis=1)
_data["hour"] = _data.apply(lambda x: x.timestamp.hour, axis=1)
_data["minute"] = _data.apply(lambda x: x.timestamp.minute, axis=1)
_data["weekday"] = _data.apply(lambda x: x.timestamp.weekday(), axis=1)

mytools.evaluate_numerical_features(_data, features)
