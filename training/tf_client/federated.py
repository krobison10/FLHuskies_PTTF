# from utils import *
import timeit

import flwr as fl
import pandas as pd
from flower_client import FlowerClient
from load_data import load_data
from tf_dnn import ALL_AIRPORTS, MyTensorflowDNN
from flwr.common.typing import NDArrays, Scalar


def client_fn(cid: str, train_loaders, test_loaders) -> FlowerClient:
    net = MyTensorflowDNN.get_model("ALL")
    trainloader = train_loaders[int(cid)]
    valloader = test_loaders[int(cid)]

    return FlowerClient(net, trainloader, valloader)


def main() -> None:
    maes = pd.DataFrame(columns=["airport", "global", "federated"], index=["airport"])

    # for airport in airports:
    global_loss = 0

    start = timeit.default_timer()

    train_loaders, test_loaders = load_data()

    num_clients = len(train_loaders)

    stop = timeit.default_timer()
    print(f"Finished Processing Airlines in {int(stop-start)} seconds")

    # FedAvg
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.5,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients // 2,
        min_available_clients=num_clients,
    )

    hist = fl.simulation.start_simulation(
        client_fn=lambda x: client_fn(x, train_loaders, test_loaders),
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=50),
        strategy=strategy,
    )

    local_loss = hist.losses_distributed[-1][1]

    local_maes = pd.DataFrame(
        {"airport": "ALL", "global": global_loss, "federated": local_loss},
        index=["airport"],
    )

    maes = pd.concat([maes, local_maes])

    # maes.to_csv("global_vs_fed_june19.csv")


if __name__ == "__main__":
    main()
