import copy
import json
import os

import matplotlib.pyplot as plt  # type: ignore
import mytools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from constants import ALL_AIRPORTS, TARGET_LABEL

torch.set_default_device("cuda")

for _airport in ALL_AIRPORTS:
    train_df, val_df = mytools.get_train_and_test_ds(_airport)

    # train-test split of the dataset
    X_train = torch.tensor(train_df.drop(columns=[TARGET_LABEL]).values, dtype=torch.float32)
    y_train = torch.tensor(train_df[TARGET_LABEL].values, dtype=torch.float32).reshape(-1, 1)
    X_test = torch.tensor(val_df.drop(columns=[TARGET_LABEL]).values, dtype=torch.float32)
    y_test = torch.tensor(val_df[TARGET_LABEL].values, dtype=torch.float32).reshape(-1, 1)

    print(X_train.shape)
    print(X_test.shape)

    # Define the model
    model = nn.Sequential(
        nn.LayerNorm(X_train.shape[1]),
        nn.Linear(X_train.shape[1], 32),
        nn.ReLU(),
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )

    # loss function and optimizer
    loss_fn = nn.L1Loss()  # mean square error
    optimizer = optim.Adam(model.parameters())

    n_epochs = 10000  # number of epochs to run

    # Hold the best model
    best_mae = np.inf  # init to infinity
    best_weights: dict = {}
    history: dict[str, list] = {"loss": [], "val_loss": []}

    for epoch in range(n_epochs):
        model.train()
        # forward pass
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        # update weights
        optimizer.step()

        # evaluate accuracy at end of each epoch
        model.eval()
        y_pred = model(X_test)
        val_mae: float = float(loss_fn(y_pred, y_test))
        print(f"Epoch {epoch}: {round(val_mae, 4)}")
        history["loss"].append(float(loss))
        history["val_loss"].append(val_mae)
        if val_mae < best_mae:
            best_mae = val_mae
            best_weights = copy.deepcopy(model.state_dict())

    # restore model and return best accuracy
    model.load_state_dict(best_weights)
    print("MSE: %.2f" % best_mae)

    plt.plot(history["loss"][: n_epochs // 10], label="loss")
    plt.plot(history["val_loss"][: n_epochs // 10], label="val_loss")
    plt.xlabel("epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.title(f"validation loss curve for {_airport}")
    plt.savefig(mytools.get_model_path(f"pytorch_dnn_{_airport}_info.png"))

    _DATA: dict[str, float] = {}
    if os.path.exists(mytools.get_model_path("pytorch_dnn_model_records.json")):
        with open(mytools.get_model_path("pytorch_dnn_model_records.json"), "r", encoding="utf-8") as f:
            _DATA.update(json.load(f))

    _DATA[_airport] = best_mae

    with open(mytools.get_model_path("pytorch_dnn_model_records.json"), "w", encoding="utf-8") as f:
        json.dump(_DATA, f, indent=4, ensure_ascii=False, sort_keys=True)
