"""
Author: Trevor Tomlin
This script reads in the airport data and 
encodes the specified them using the OrdinalEncoder.
It then saves the OrdinalEncoders to a pickle file in the form of a dictionary.
"""

import pickle
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder  # type: ignore

DATA_DIRECTORY = Path("full_tables/")
ROOT = Path("/home/ydlin/FLHuskies_PTTF/")


def main() -> None:
    # DATA_DIR = Path("/home/ydlin/FL_Huskies_PTTF/train_tables/")

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
        "gufi_flight_number",
        "gufi_flight_major_carrier",
        "gufi_flight_destination_airport",
        "aircraft_engine_class",
        "aircraft_type",
        "major_carrier",
        "flight_type",
    ]

    unique = defaultdict(set)

    Encoder = partial(OrdinalEncoder, handle_unknown="use_encoded_value", unknown_value=-1)
    encoders = defaultdict(Encoder)

    for airport in airports:
        print(f"Processing Airport {airport}")
        data = pd.read_csv(f"data_apr16/tables/full_tables/{airport}_full.csv").sort_values("timestamp")
        data["precip"] = data["precip"].astype(str)
        data["isdeparture"] = data["isdeparture"].astype(str)

        for column in encoded_columns:
            unique[column].update(data[column].unique())

    for column in encoded_columns:
        encoders[column].fit(np.array(list(unique[column])).reshape(-1, 1))

    with open("encoders.pickle", "wb") as f:
        pickle.dump(encoders, f)


if __name__ == "__main__":
    main()
