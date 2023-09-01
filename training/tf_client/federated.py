# from utils import *
import timeit

import flwr as fl
import numpy
from flwr.common import Metrics
from matplotlib.pylab import plt

from .flower_client import FlowerClient
from .load_data import load_all_airports
from .mytools import get_model_path
from .tf_dnn import MyTensorflowDNN


def client_fn(cid: str, train_loaders, test_loaders) -> FlowerClient:
    net = MyTensorflowDNN.get_model("ALL")
    trainloader = train_loaders[int(cid)]
    valloader = test_loaders[int(cid)]

    return FlowerClient(net, trainloader, valloader)


def train():
    # maes = pd.DataFrame(columns=["airport", "global", "federated"], index=["airport"])

    # client_resources = None
    # physical_devices = tf.config.list_physical_devices("GPU")
    # if len(physical_devices) > 0:
    #    client_resources = {"num_gpus": 1}
    start = timeit.default_timer()

    # Load ALL airports, modified for training to include all data in train_loaders
    train_loaders, test_loaders = load_all_airports()

    num_clients = len(train_loaders)

    stop = timeit.default_timer()
    print(f"Finished Processing {num_clients} Airlines in {int(stop-start)} seconds")

    server_model = MyTensorflowDNN.get_model("ALL")
    params = server_model.get_weights()

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

    hist: fl.server.history = fl.simulation.start_simulation(
        client_fn=lambda x: client_fn(x, train_loaders, test_loaders),
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=100),
        strategy=strategy,
    )

    print(hist)

    # Plot and label the training and validation loss values
    plt.plot(numpy.transpose(hist.losses_distributed)[1], label="losses distributed")
    plt.plot(numpy.transpose(hist.metrics_distributed["loss"])[1], label="metrics distributed")
    plt.legend(loc="lower right")

    # Add in a title and axes labels
    plt.title("Flower Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    # save model
    server_model.save(get_model_path("tf_dnn_global_model"))
    # save loss image
    plt.savefig(get_model_path("tf_dnn_global_loss.png"))
    print("Saved server model")


def fit_config(server_round: int):
    config = {
        "server_round": server_round,  # The current round of federated learning
        "local_epochs": 5 if server_round < 3 else 6,  #
    }
    return config


def weighted_average(metrics: list[tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"loss": sum(accuracies) / sum(examples)}
