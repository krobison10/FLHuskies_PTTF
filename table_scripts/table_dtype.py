#
# Author: Yudong Lin
#
# A simple class for keep tracking of typing for output table
#

import pandas as pd


class TableDtype:
    INT_COLUMNS: tuple[str, ...] = (
        "minutes_until_pushback",
        "minutes_until_etd",
        "temperature",
        "wind_direction",
        "wind_speed",
        "wind_gust",
        "gufi",
        "estdep_next_30min",
        "estdep_next_60min",
        "estdep_next_180min",
        "estdep_next_360min",
    )

    FLOAT_COLUMNS: tuple[str, ...] = ("cloud_ceiling", "visibility")

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
    @classmethod
    def fix_potential_missing_int_features(cls, _df: pd.DataFrame) -> pd.DataFrame:
        for _col in cls.INT_COLUMNS:
            if _col in _df.columns:
                _df[_col] = _df[_col].fillna(0).astype(int)

        return _df
