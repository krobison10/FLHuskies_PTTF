#
# Authors:
# - Kyler Robison
# - Yudong Lin
#
# A set of helper functions
#

import os
import pandas as pd


# get a valid path for a csv file
# try to return the path for uncompressed csv file first
# if the uncompressed csv does not exists, then return the path for compressed csv file
def get_csv_path(*argv: str) -> str:
    etd_csv_path: str = os.path.join(*argv)
    if not os.path.exists(etd_csv_path):
        etd_csv_path += ".bz2"
    return etd_csv_path


# specify an airport to split for only one, otherwise a split for all airports will be executed
def train_test_split(table: pd.DataFrame, ROOT: str, airport: str | None = None) -> None:
    valdata = pd.read_csv(os.path.join(ROOT, "_data", "submission_format.csv"))

    # If there is a specific airport then we are only interested in those rows
    ext: str = "ALL"
    if airport is not None:
        ext = f"{airport}"
        valdata = valdata[valdata.airport == airport]

    mygufis = valdata.gufi.unique()
    testdata = table[table.gufi.isin(mygufis)]
    traindata = table[~table.gufi.isin(mygufis)]

    # replace these paths with any desired ones if necessary
    traindata.sort_values(["gufi", "timestamp"]).to_csv(
        os.path.join(ROOT, "train_tables", f"{ext}_train.csv"), index=False
    )
    testdata.sort_values(["gufi", "timestamp"]).to_csv(
        os.path.join(ROOT, "validation_tables", f"{ext}_validation.csv"), index=False
    )
