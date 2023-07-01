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
from load_data import load_airports

def train_global(df):
    start = timeit.default_timer()
    X = df[features]
    y = df["minutes_until_pushback"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for column in encoded_columns:
        try:
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
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

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32)

    net = Net().to(DEVICE)

    train(net, train_loader, 5, True)
    test_loss = test(net, test_loader)
    print(f"Global Model Loss: {test_loss}")
    stop = timeit.default_timer()
    print(f"Finished Training Global Model in {int(stop-start)} seconds")
    return test_loss

def main():
    maes = pd.DataFrame(columns=["airport", "global", "federated"], index=["airport"])

    #for airport in airports:
    for airport in airports[-1:]:    

        global_loss = 0

        start = timeit.default_timer()

        train_loaders, test_loaders = load_airports(airport)

        num_clients = len(train_loaders)

        stop = timeit.default_timer()
        print(f"Finished Processing Airlines in {int(stop-start)} seconds")

        server_model = Net().to(DEVICE)

        # FedAvg
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=0.5,
            min_fit_clients=num_clients,
            min_evaluate_clients=num_clients//2,
            min_available_clients=num_clients,
            evaluate_fn=get_evaluate_fn(server_model)
        )


        hist = fl.simulation.start_simulation(
            client_fn=lambda x: client_fn(x, train_loaders, test_loaders),
            num_clients=num_clients,
            config=fl.server.ServerConfig(num_rounds=5),
            strategy=strategy,
            client_resources=client_resources,
        )

        local_loss = hist.losses_distributed[-1][1]

        local_maes = pd.DataFrame({"airport":airport, "global":global_loss, "federated":local_loss}, index=["airport"])

        maes = pd.concat([maes, local_maes])

    #maes.to_csv("global_vs_fed_june19.csv")


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