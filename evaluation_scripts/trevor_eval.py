#
# Kyler Robison
#
# Evaluates Trevor's model trained on individual airports using all current features
#

import os
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

from sklearn.metrics import mean_absolute_error

# this table is produced simply by running train_test_split on the main table
val_df = pd.read_csv(os.path.join("..", "validation_tables", "master_validation.csv"))

with open(os.path.join("..", "models", "lgbm_etd_mfs_lamp.pickle"), "rb") as fp:
    models = pickle.load(fp)

with open(os.path.join("..", "models", "mfs_lamp_encoders.pickle"), "rb") as fp:
    encoders = pickle.load(fp)


print(f"Encoding columns: {list(dict.keys(encoders))}")
for encode_col in encoders:
    if encode_col == "airport":
        continue
    if encode_col == "precip":
        val_df[[encode_col]] = encoders[encode_col].transform(val_df[[encode_col]].values.astype(str))
    else:
        val_df[[encode_col]] = encoders[encode_col].transform(val_df[[encode_col]].values)


input_features = [
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

airport_tables = []

for airport in airports:
    airport_input = val_df.groupby("airport").get_group(airport).copy()
    y_pred = models[airport].predict(airport_input[input_features]).clip(min=0, max=np.inf)
    airport_input["minutes_until_pushback_pred"] = y_pred
    airport_tables.append(airport_input)

val_df = pd.concat(airport_tables, ignore_index=True)

print("\nMAE: " + str(mean_absolute_error(val_df["minutes_until_pushback"], val_df["minutes_until_pushback_pred"])))
