#
# Author: Yudong Lin
#
# extract potential useful features from gufi
#


import pandas as pd  # type: ignore


def extract_and_add_gufi_features(_df: pd.DataFrame) -> pd.DataFrame:
    """
    def split_gufi(gufi:str):
        information:list = gufi.split(".")

        gufi_flight_number:str = information[0]
        gufi_flight_destination :str = information[2]
        gufi_flight_destination

    _df[['C','D']] = df.apply(lambda x: pd.Series([x['B'] + 'k', x['B'] + 'n']), axis=1)

    _df["FAA_system"] = _df.apply(lambda x: "TFM" if x.gufi.endswith("TFM") else "TFM_TFDM" if x.gufi.endswith("TFM_TFDM") else "OTHER", axis=1)
    """

    return _df
