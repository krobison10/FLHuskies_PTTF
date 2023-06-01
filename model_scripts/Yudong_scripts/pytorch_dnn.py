import copy
import os

import mytools
import numpy as np
import torch
from constants import ALL_AIRPORTS, TARGET_LABEL

torch.set_default_device("cuda")

# Create an empty model
model = torch.nn.Sequential(
    torch.nn.LayerNorm(44),
    torch.nn.Linear(44, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 1),
)
print("----------------------------------------")
if os.path.exists(mytools.get_model_path("pytorch_dnn_model.pt")):
    print("A existing model has been found and will be loaded.")
    model.load_state_dict(torch.load(mytools.get_model_path("pytorch_dnn_model.pt")))
else:
    print("Creating new model.")
print("----------------------------------------")

# loss function and optimizer
loss_fn = torch.nn.L1Loss()  # mean square error
optimizer = torch.optim.Adam(model.parameters())

# number of epochs to run
n_epochs = 10000

# update database name
mytools.ModelRecords.set_name("pytorch_dnn_model_records")

for _airport in ALL_AIRPORTS:
    train_df, val_df = mytools.get_train_and_test_ds(_airport)

    X_train: torch.Tensor = torch.as_tensor(train_df.drop(columns=[TARGET_LABEL]).values, dtype=torch.float32)
    y_train: torch.Tensor = torch.as_tensor(train_df[TARGET_LABEL].values, dtype=torch.float32).reshape(-1, 1)
    X_test: torch.Tensor = torch.as_tensor(val_df.drop(columns=[TARGET_LABEL]).values, dtype=torch.float32)
    y_test: torch.Tensor = torch.as_tensor(val_df[TARGET_LABEL].values, dtype=torch.float32).reshape(-1, 1)

    # Hold the best model
    best_mae: float = np.inf  # init to infinity
    best_weights: dict = model.state_dict()
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
        print(f"$({_airport}) Epoch {epoch}: {round(val_mae, 4)}")
        history["loss"].append(float(loss))
        history["val_loss"].append(val_mae)
        if val_mae < best_mae:
            best_mae = val_mae
            best_weights = copy.deepcopy(model.state_dict())

    # restore model and return best accuracy
    model.load_state_dict(best_weights)
    torch.save(model.state_dict(), mytools.get_model_path("pytorch_dnn_model.pt"))
    print("MSE: %.2f" % best_mae)

    # save history
    mytools.plot_history(_airport, history, f"pytorch_dnn_{_airport}_info.png")
    mytools.ModelRecords.update(_airport, "best_mae", best_mae, True)
