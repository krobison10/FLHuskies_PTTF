# Daniil Filienko
#
# Running the model emsemble with Kyler's Train Split for Trevor
# to attain accuracy values for individual airports and overall
from Yudong_scripts.mytools import *
import matplotlib.pyplot as plt
from train_test_split import *
import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns
from lightgbm import LGBMRegressor, Dataset
import lightgbm as lgb
import pickle
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_absolute_error
import argparse
from pathlib import Path

# ---------------------------------------- MAIN ----------------------------------------
DATA_DIRECTORY = Path("full_tables")
OUTPUT_DIRECTORY = Path("./models/Daniil_models")

airports = [    
    "KATL",
    "KCLT",
    "KDEN",
    "KDFW",
    "KJFK",
    "KMEM",
    "KMIA",
    "KORD",
    "KPHX",
    "KSEA",
]
parser: argparse.ArgumentParser = argparse.ArgumentParser()
parser.add_argument("-c", help="which column to look up unique values for")
args: argparse.Namespace = parser.parse_args()

column_name: str = "gufi_flight_major_carrier" if args.c is None else str(args.c)


total_airlines = set([])

for airport in airports:
    # replace this path with the locations of the full tables for each airport if necessary
    df = pd.read_csv(DATA_DIRECTORY / f"{airport}_full.csv",parse_dates=["timestamp"])

    # get the unique values in the column
    unique_values = df[column_name].unique()

    # count the number of occurrences of each unique value
    value_counts = df[column_name].value_counts()
    
    print(f"Listing Airlines for {airport}\n")

    # print the results
    print(f"Number of unique airlines based on major carrier: {len(unique_values)}\n")
    for value in unique_values:
        count = value_counts.get(value)
        print(f"{value}: {count} rows")
        total_airlines.add(value)

    
print(f"Total number of unique values in {column_name}: {len(total_airlines)}")

exit()