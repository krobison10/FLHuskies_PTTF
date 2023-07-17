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
from typing import List, Tuple, Dict, Optional, Callable, Union
import matplotlib.pyplot as plt
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
from load_data import load_airports, load_all_airports
import numpy as np


from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg


def main():
    maes = pd.DataFrame(columns=["airport", "global", "federated"], index=["airport"])

    start = timeit.default_timer()

    # Load ALL airports, modified for training to include all data in train_loaders
    train_loaders, test_loaders = load_all_airports()

    num_clients = len(train_loaders)

    stop = timeit.default_timer()
    print(f"Finished Processing {num_clients} Airlines in {int(stop-start)} seconds")

    server_model = Net().to(DEVICE)
    params = get_parameters(server_model)

    # FedAvg
    strategy = fl.server.strategy.FedMedian(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
        # evaluate_fn=get_evaluate_fn(server_model, test_loaders),
        initial_parameters=fl.common.ndarrays_to_parameters(params),
        on_fit_config_fn=fit_config,
    )

    hist = fl.simulation.start_simulation(
        client_fn=lambda x: client_fn(x, train_loaders, test_loaders),
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
        client_resources=client_resources,
    )

    # torch.save(server_model, "model.pt")
    # print("Saved server model")


def client_fn(cid: str, train_loaders, test_loaders) -> FlowerClient:
    net = Net().to(DEVICE)
    trainloader = train_loaders[int(cid)]
    valloader = test_loaders[int(cid)]

    return FlowerClient(net, trainloader, valloader)


def get_evaluate_fn(model, test_loaders):
    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    # Use the validation set to have the default values

    def evaluate(
        server_round: int, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        model = Net().to(DEVICE)
        set_parameters(model, parameters)  # Update model with the latest parameters
        losses = 0
        accuracies = 0

        for i in range(len(test_loaders)):
            losses = losses + test(model, test_loaders[i])

        loss = losses / len(test_loaders)
        print(f"Server-side evaluation loss {loss} / accuracy {loss}")
        torch.save(model, f"model_{server_round}.pt")

        print(f"Saved the model, round {server_round}")
        return float(loss), {"accuracy": float(loss)}

    return evaluate


def fit_config(server_round: int):
    config = {
        "server_round": server_round,  # The current round of federated learning
        "local_epochs": 5 if server_round < 3 else 6,  #
    }
    return config


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


if __name__ == "__main__":
    main()
