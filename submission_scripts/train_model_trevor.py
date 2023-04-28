
#from utils import *
import os
import pandas as pd
import pickle
import numpy as np
from pathlib import Path
from lightgbm import LGBMRegressor, Dataset
import lightgbm as lgb
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
#import seaborn as sns
from datetime import timedelta
from pandarallel import pandarallel
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import multiprocessing as mp
from collections import defaultdict

# ---------------------------------------- MAIN ----------------------------------------
DATA_DIRECTORY = Path("data_apr16/tables/full_tables/")
ROOT1 = Path("/home/ydlin/FLHuskies_PTTF/")
ROOT = Path("/home/ttomlin/FLHuskies_PTTF")

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

encoded_columns = [
    "airport",
    "departure_runways",
    "arrival_runways",
    "cloud",
    "lightning_prob",
    "precip",
    "gufi_flight_major_carrier",
    "gufi_flight_destination_airport",
    "aircraft_engine_class",
    "aircraft_type",
    "major_carrier",
    "flight_type",
    ]

features = [
    "gufi_flight_major_carrier",
    "deps_3hr",
    "deps_30hr",
    "arrs_3hr",
    "arrs_30hr",
    "deps_taxiing",
    "arrs_taxiing",
    "exp_deps_15min",
    "exp_deps_30min",
    "standtime_30hr",
    "dep_taxi_30hr",
    "arr_taxi_30hr",
    "minute",
    "gufi_flight_destination_airport",
    "month",
    "day",
    "hour",
    "year",
    "weekday",
    "airport",
    "minutes_until_etd",
    "aircraft_engine_class",
    "aircraft_type",
    "major_carrier",
    "flight_type",
    "temperature",
    "wind_direction",
    "wind_speed",
    "wind_gust",
    "cloud_ceiling",
    "visibility",
    "cloud",
    "lightning_prob",
    "precip",
]

models = defaultdict(LGBMRegressor)

for airport in airports:
    print(f"Processing Airport {airport}")
    df = pd.read_csv(os.path.join(ROOT, DATA_DIRECTORY, f"{airport}_full.csv"), parse_dates=["timestamp"])

    df["precip"] = df["precip"].astype(str)
    df["isdeparture"] = df["isdeparture"].astype(str)

    with (Path("encoders.pickle")).open("rb") as fp:
        encoders = pickle.load(fp)

    for col in encoded_columns:
        df[[col]] = encoders[col].transform(df[[col]].values)

    X_train = (df[features])
    y_train = (df["minutes_until_pushback"])

    train_data = lgb.Dataset(X_train, label=y_train)

    params = {
            'objective': 'regression_l1',
            'metric': 'mae',
            "num_leaves":4096,
            "n_estimators":128,
            #"learning_rate":0.05,
            }

    gbm = lgb.train(params, train_data)

    models[airport] = gbm

with open("models.pickle", "wb") as f:
        pickle.dump(models, f)

exit()
