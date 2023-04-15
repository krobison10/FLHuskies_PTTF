
from utils import *
import pandas as pd
import pickle
import numpy as np
from pathlib import Path
from lightgbm import LGBMRegressor, Dataset
import lightgbm as lgb
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from pandarallel import pandarallel
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import multiprocessing as mp
from collections import defaultdict

# ---------------------------------------- MAIN ----------------------------------------
DATA_DIRECTORY = Path("data_apr14/tables/full_tables/")
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

models = defaultdict(LGBMRegressor)

for airport in airports:
    print(f"Processing Airport {airport}")
    df = pd.read_csv(os.path.join(ROOT, DATA_DIRECTORY, f"{airport}_full.csv"), parse_dates=["timestamp"])

    offset = 2
    features_all = (df.columns.values.tolist())[offset:(len(df.columns.values))]
    features_remove = ("gufi_flight_date","minutes_until_pushback")
    features = [x for x in features_all if x not in features_remove and not str(x).startswith("feat_lamp_") and not str(x).startswith("feats_lamp_")]

    X_train = (df[features])
    y_train = (df["minutes_until_pushback"])

    for c in tqdm(X_train.columns):
        col_type = X_train[c].dtype
        if col_type == 'object' or col_type == 'string' or "cat" in c:
            X_train[c] = X_train[c].astype('category')

    gbm = LGBMRegressor(objective="regression_l1",
                        num_leaves=128,
                        n_estimators=128,
                        #learning_rate=0.05,
                        )

    gbm.fit(X_train, y_train)

    models[airport] = gbm

with open("models.pickle", "wb") as f:
        pickle.dump(models, f)

exit()
