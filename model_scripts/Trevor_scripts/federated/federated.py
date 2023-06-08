# from utils import *
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
import seaborn as sns
from datetime import timedelta
from pandarallel import pandarallel
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import multiprocessing as mp
from collections import defaultdict
from collections import OrderedDict
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import timeit
from sklearn.model_selection import train_test_split
import flwr as fl
from flwr.common import Metrics
from torch.utils.data import Dataset
from config import *
from flower_client import *
from airline_dataset import *
from net import *
from train_test import *
from flwr.common.typing import NDArray, NDArrays, Parameters, Scalar, Optional


def train_global(df):
    start = timeit.default_timer()
    X = df[features]
    y = df["minutes_until_pushback"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # for column in tqdm(encoded_columns):
    for column in encoded_columns:
        try:
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            # print(train_df[column])
            # print(train_df[[column]])
            X_train[column] = enc.fit_transform(X_train[[column]])
            X_test[column] = enc.transform(X_test[[column]])
        except Exception as e:
            print(e)
            print(column)
            exit()

    X_train = X_train.to_numpy(dtype=np.float32, copy=True)
    X_test = X_test.to_numpy(dtype=np.float32, copy=True)
    y_train = y_train.to_numpy(dtype=np.float32, copy=True).reshape(-1, 1)
    y_test = y_test.to_numpy(dtype=np.float32, copy=True).reshape(-1, 1)

    train_data = AirlineDataset(X_train, y_train)
    test_data = AirlineDataset(X_test, y_test)

    # print(len(train_data))
    # print(len(test_data))

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32)

    net = Net().to(DEVICE)

    train(net, train_loader, 1, True)
    print(f"Global Model Loss: {test(net, test_loader)}")
    stop = timeit.default_timer()
    print(f"Finished Training Global Model in {int(stop-start)} seconds")


def main():
    # Airline : df
    dfs = defaultdict(pd.DataFrame)

    start = timeit.default_timer()

    for airport in airports[:1]:
        print(f"Processing Airport {airport}")
        # df = pd.read_csv(os.path.join(ROOT1, DATA_DIRECTORY, f"{airport}_full.csv"), parse_dates=["timestamp"])
        df = pd.read_csv(
            os.path.join(ROOT, "data_apr16/tables/train_tables/", f"{airport}_train.csv"), parse_dates=["timestamp"]
        )
        df["precip"] = df["precip"].astype(str)
        df["isdeparture"] = df["isdeparture"].astype(str)

        df = df[df["gufi_flight_major_carrier"].isin(AIRLINES)]

        train_global(df)

        airlines = [(x, pd.DataFrame(y)) for x, y in df.groupby("gufi_flight_major_carrier", as_index=False)]

        for airline, df in airlines:
            dfs[airline] = pd.concat([dfs[airline], df])

        # break

    stop = timeit.default_timer()
    print(f"Finished Processing Airports in {int(stop-start)} seconds")

    num_clients = len(dfs)

    # print(dfs.keys())

    start = timeit.default_timer()

    train_loaders, test_loaders = load_datasets(dfs)

    stop = timeit.default_timer()
    print(f"Finished Processing Airlines in {int(stop-start)} seconds")

    server_model = Net().to(DEVICE)

    # FedAvg
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.5,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients // 2,
        min_available_clients=num_clients,
        evaluate_fn=get_evaluate_fn(server_model),
    )

    fl.simulation.start_simulation(
        client_fn=lambda x: client_fn(x, train_loaders, test_loaders),
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=1),
        strategy=strategy,
        client_resources=client_resources,
    )

    with open("server.pickle", "wb") as f:
        pickle.dump(server_model.state_dict(), f)


def load_datasets(dfs):
    train_loaders = []
    test_loaders = []

    for airline, df in dfs.items():
        print(f"Processing Airline {airline}")
        # df = dfs[airline]

        # print(df.head())

        # train_df, val_df = train_test_split(table=df, ROOT=ROOT1, airport="", save=False)

        # print(X_train.head())

        # X_train = train_df[features].to_numpy()
        # X_test = val_df[features].to_numpy()
        # y_train = train_df["minutes_until_pushback"].to_numpy()
        # y_test = val_df["minutes_until_pushback"].to_numpy()

        X = df[features]
        y = df["minutes_until_pushback"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # for column in tqdm(encoded_columns):
        for column in encoded_columns:
            try:
                enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                # print(train_df[column])
                # print(train_df[[column]])
                X_train[column] = enc.fit_transform(X_train[[column]])
                X_test[column] = enc.transform(X_test[[column]])
            except Exception as e:
                print(e)
                print(column)
                exit()

        X_train = X_train.to_numpy(dtype=np.float32, copy=True)
        X_test = X_test.to_numpy(dtype=np.float32, copy=True)
        y_train = y_train.to_numpy(dtype=np.float32, copy=True).reshape(-1, 1)
        y_test = y_test.to_numpy(dtype=np.float32, copy=True).reshape(-1, 1)

        X_train.flags.writeable = True
        y_train.flags.writeable = True
        X_test.flags.writeable = True
        y_test.flags.writeable = True

        # X_train = train_df[features].to_numpy(dtype=np.float32)
        # X_test = val_df[features].to_numpy(dtype=np.float32)
        # y_train = train_df["minutes_until_pushback"].to_numpy(dtype=np.float32).reshape(-1, 1)
        # y_test = val_df["minutes_until_pushback"].to_numpy(dtype=np.float32).reshape(-1, 1)

        train_data = AirlineDataset(X_train, y_train)
        test_data = AirlineDataset(X_test, y_test)

        # print(len(train_data))
        # print(len(test_data))

        train_loaders.append(DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True))
        test_loaders.append(DataLoader(test_data, batch_size=BATCH_SIZE))

    return train_loaders, test_loaders


def client_fn(cid: str, train_loaders, test_loaders) -> FlowerClient:
    net = Net().to(DEVICE)
    trainloader = train_loaders[int(cid)]
    valloader = test_loaders[int(cid)]

    return FlowerClient(net, trainloader, valloader)


def get_evaluate_fn(model):
    def evaluate(
        server_round: int, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        return 0.0, {"accuracy": 0.0}

    return evaluate


if __name__ == "__main__":
    main()
