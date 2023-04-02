#
# Author: Yudong Lin
#
# A simple class for keep tracking of typing for output table
#

import pandas as pd  # type: ignore


class TableDtype:
    INT_COLUMNS: tuple[str, ...] = ("minutes_until_pushback", "minutes_until_etd", "temperature", "wind_direction", "wind_speed", "wind_gust")

    FLOAT_COLUMNS: tuple[str, ...] = ("delay_3hr", "delay_30hr", "standtime_3hr", "standtime_30hr", "cloud_ceiling", "visibility")

    STR_COLUMS: tuple[str, ...] = (
        "cloud",
        "lightning_prob",
        "precip",
        "departure_runway_actual",
        "aircraft_engine_class",
        "aircraft_type",
        "major_carrier",
        "flight_type",
        "gufi_end_label",
    )

    # fill potential missing int features with 0
    def fix_potential_missing_int_features(_df: pd.DataFrame) -> pd.DataFrame:
        columns_need_normalize: tuple[str, ...] = (
            "delay_3hr",
            "delay_30hr",
            "standtime_3hr",
            "standtime_30hr",
            "temperature",
            "wind_direction",
            "wind_speed",
            "wind_gust",
            "cloud_ceiling",
            "visibility",
        )

        float_columns: tuple[str, ...] = ("delay_3hr", "delay_30hr", "standtime_3hr", "standtime_30hr")

        for _col in columns_need_normalize:
            _df[_col] = _df[_col].fillna(0)
            if _col in float_columns:
                _df[_col] = _df[_col].round(10)

        return _df
