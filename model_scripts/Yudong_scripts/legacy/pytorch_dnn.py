import copy

import matplotlib.pyplot as plt  # type: ignore
import mytools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from constants import TARGET_LABEL

airport = "KMEM"

train_df, val_df = mytools.get_train_and_test_ds(airport)

torch.set_default_device("cuda")

# train-test split of the dataset
X_train = torch.tensor(train_df.drop(columns=[TARGET_LABEL]).values, dtype=torch.float32)
y_train = torch.tensor(train_df[TARGET_LABEL].values, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(val_df.drop(columns=[TARGET_LABEL]).values, dtype=torch.float32)
y_test = torch.tensor(val_df[TARGET_LABEL].values, dtype=torch.float32).reshape(-1, 1)

print(X_train.shape)
print(X_test.shape)

# Define the model
model = nn.Sequential(
    nn.Linear(44, 64),
    nn.ReLU(inplace=True),
    nn.Linear(64, 64),
    nn.ReLU(inplace=True),
    nn.Linear(64, 128),
    nn.ReLU(inplace=True),
    nn.Linear(128, 64),
    nn.ReLU(inplace=True),
    nn.Linear(64, 32),
    nn.ReLU(inplace=True),
    nn.Linear(32, 16),
    nn.ReLU(inplace=True),
    nn.Linear(16, 1),
)

# loss function and optimizer
loss_fn = nn.L1Loss()  # mean square error
optimizer = optim.Adam(model.parameters(), lr=0.0001)

n_epochs = 10000  # number of epochs to run
batch_size = 10  # size of each batch
batch_start = torch.arange(0, len(X_train), batch_size)

# Hold the best model
best_mae = np.inf  # init to infinity
best_weights = {}
history = []

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
    mae = loss_fn(y_pred, y_test)
    mae = float(mae)
    print(f"Epoch {epoch}: {mae}")
    history.append(mae)
    if mae < best_mae:
        best_mae = mae
        best_weights = copy.deepcopy(model.state_dict())

# restore model and return best accuracy
model.load_state_dict(best_weights)
print("MSE: %.2f" % best_mae)
print("RMSE: %.2f" % np.sqrt(best_mae))
plt.plot(history)
plt.show()
