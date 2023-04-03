# Takes original train_labels file for an airport (SEA is used below)
# Adds 2 additional columns:
# - departure_runway_estimated time (ETD)
# - minutes from the current time stamp until this ETD

from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_DIRECTORY = Path("data/")
airport = "KSEA"

#pushback = pd.read_csv(DATA_DIRECTORY / f"train_labels_{airport}.csv.bz2", parse_dates=["timestamp"])
#print(pushback)

pushback = pd.read_csv(DATA_DIRECTORY / f"prescreened_train_labels_{airport}.csv.bz2", parse_dates=["timestamp"])

lamp = pd.read_csv(DATA_DIRECTORY / airport / f"{airport}_lamp.csv.bz2", parse_dates=["forecast_timestamp", "timestamp"])

lamp.sort_values("timestamp", inplace=True)

etd = pd.read_csv(DATA_DIRECTORY / airport / f"{airport}_lmp.csv.bz2", parse_dates=["departure_runway_estimated_time", "timestamp"])

etd.sort_values("timestamp", inplace=True)

data = pushback

times = pd.to_datetime(data.timestamp.unique())
print("Started building table at ", pd.Timestamp.now())

bigtable = pd.DataFrame()
for mytime in times:
    now_data = data.loc[data.timestamp == mytime].reset_index(drop=True)
    now_etd = etd.loc[(etd.timestamp > mytime - timedelta(hours=30)) & (etd.timestamp <= mytime)]
    latest_now_etd = now_etd.groupby("gufi").last().departure_runway_estimated_time
    updated_data = now_data.merge(latest_now_etd, how="left", on="gufi")
    updated_data["minutes_until_departure"] = (updated_data.departure_runway_estimated_time - updated_data.timestamp).dt.total_seconds() / 60
    bigtable = pd.concat([bigtable,updated_data],ignore_index=True)
   
bigtable.to_csv("bigtable.csv")
print("Finished building table at ", pd.Timestamp.now())  
