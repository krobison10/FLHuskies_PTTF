#
# Author: Yudong Lin
#
# extract potential useful features from gufi
#

from datetime import datetime

import pandas as pd  # type: ignore


def extract_and_add_gufi_features(_df: pd.DataFrame) -> pd.DataFrame:
    def split_gufi(gufi: str):
        information: list = gufi.split(".")
        gufi_flight_number: str = information[0]
        gufi_flight_destination_airport: str = information[2]
        gufi_flight_date: datetime = datetime.strptime("_".join((information[3], information[4], information[5][:2])), "%y%m%d_%H%M_%S")
        gufi_flight_FAA_system: str = information[6]
        return pd.Series([gufi_flight_number, gufi_flight_destination_airport, gufi_flight_date, gufi_flight_FAA_system])

    _df[["gufi_flight_number", "gufi_flight_destination_airport", "gufi_flight_date", "gufi_flight_FAA_system"]] = _df.apply(
        lambda x: split_gufi(x["gufi"]), axis=1
    )

    return _df
