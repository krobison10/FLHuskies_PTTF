#
# Author: Trevor Tomlin
#
# Add mfs information to the data frame
#

import pandas as pd  # type: ignore


def add_mfs(_df: pd.DataFrame, mfs_csv_file_path: str) -> pd.DataFrame:
    mfs_df = pd.read_csv(mfs_csv_file_path, dtype={"major_carrier": str})
    return _df.merge(mfs_df, how="left", on="gufi")
