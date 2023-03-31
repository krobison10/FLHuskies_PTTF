# M. De Cock; Mar 24, 2023
# Modified by Kyler Robison
# Splits a table into train and test based on a list of GUFIs
import os
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")

# specify an airport to split for only one, otherwise a split for all airports will be executed
def split(table: pd.DataFrame, airport: str = None):
    valdata = pd.read_csv(os.path.join(ROOT, "_data", "submission_format.csv"))

    # If there is a specific airport then we are only interested in those rows
    ext = ""
    if airport:
        ext = f"{airport}_"
        valdata = valdata[valdata.airport == airport]

    mygufis = valdata.gufi.unique()
    testdata = table[table.gufi.isin(mygufis)]
    traindata = table[~table.gufi.isin(mygufis)]

    traindata.to_csv(os.path.join(ROOT, "train_tables", f"{ext}train.csv"), index=False)
    testdata.to_csv(os.path.join(ROOT, "validation_tables", f"{ext}validation.csv"), index=False)


if __name__ == "__main__":
    airport = "KSEA"
    df = pd.read_csv(os.path.join(ROOT, "train_tables", f"{airport}_full.csv"))
    split(table=df, airport=airport)
