"""
Author: Trevor Tomlin
This script reads in the airport data and 
encodes the specified them using the OrdinalEncoder.
It then saves the OrdinalEncoders to a pickle file in the form of a dictionary.
"""

from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from functools import partial
from pathlib import Path
import pickle

def main():
    DATA_DIR = Path("/home/ydlin/FL_Huskies_PTTF/train_tables/")

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

    encoded_columns = ["airport", "aircraft_engine_class", "aircraft_type", "major_carrier", "flight_type"]

    unique = defaultdict(set)

    encoders = defaultdict(OrdinalEncoder)

    for airport in airports:
        data = pd.read_csv(f"/home/ydlin/FLHuskies_PTTF/main_{airport}_prescreened.csv").sort_values("timestamp")

        for column in encoded_columns:

            unique[column].update(data[column].unique())

    for column in encoded_columns:
        encoders[column].fit(np.array(list(unique[column])).reshape(-1, 1))

    for airport in airports:
        data = pd.read_csv(f"/home/ydlin/FLHuskies_PTTF/main_{airport}_prescreened.csv").sort_values("timestamp")
        for column in encoded_columns:
            data[column] = encoders[column].transform(data[column].values.reshape(-1, 1)).astype(int)

        data.to_csv(f"{airport}_etd_w_mfs_encoded.csv", index=False)

    #with open("mfs_encoders.pickle", "wb") as f:
    #    pickle.dump(encoders, f)

if __name__ == "__main__":
    main()