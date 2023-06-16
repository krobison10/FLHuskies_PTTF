import copy
import os

import mytools
import numpy as np
import torch
from constants import TARGET_LABEL
from sklearn.preprocessing import Normalizer  # type: ignore

torch.set_default_device("cuda")


class MyTorchDNN:
    NUM_OF_FEATURES = -1

    @classmethod
    def __get_model_path(cls, _airport: str) -> str:
        return mytools.get_model_path(f"pytorch_dnn_{_airport}_model.pt")

    @classmethod
    def get_model(cls, _airport: str, load_if_exists: bool = True) -> torch.nn.Sequential:
        # Create an empty model
        _model: torch.nn.Sequential = torch.nn.Sequential(
            # torch.nn.LayerNorm(cls.NUM_OF_FEATURES),
            torch.nn.Linear(cls.NUM_OF_FEATURES, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )

        _model_path: str = cls.__get_model_path(_airport)
        print("----------------------------------------")
        if load_if_exists is True and os.path.exists(_model_path):
            print("A existing model has been found and will be loaded.")
            _model.load_state_dict(torch.load(_model_path))
        else:
            print("Creating new model.")
        print("----------------------------------------")

        return _model

    @classmethod
    def train(cls, _airport: str) -> None:
        # update database name
        mytools.ModelRecords.set_name("pytorch_dnn_model_records")

        # load train and test data frame
        train_df, val_df = mytools.get_train_and_test_ds(_airport)

        X_train_nd: np.ndarray = train_df.drop(columns=[TARGET_LABEL]).values
        X_test_nd: np.ndarray = val_df.drop(columns=[TARGET_LABEL]).values

        transformer: Normalizer = Normalizer()
        transformer = transformer.fit(X_train_nd)
        transformer = transformer.fit(X_test_nd)
        X_train_nd = transformer.transform(X_train_nd)
        X_test_nd = transformer.transform(X_test_nd)

        X_train: torch.Tensor = torch.as_tensor(X_train_nd)
        y_train: torch.Tensor = torch.as_tensor(train_df[TARGET_LABEL].values, dtype=torch.int16).reshape(-1, 1)
        X_test: torch.Tensor = torch.as_tensor(X_test_nd)
        y_test: torch.Tensor = torch.as_tensor(val_df[TARGET_LABEL].values, dtype=torch.int16).reshape(-1, 1)

        cls.NUM_OF_FEATURES = X_train.shape[1]

        model = cls.get_model(_airport)

        # Hold the best model
        best_mae: float = np.inf  # init to infinity
        history: dict[str, list] = {"loss": [], "val_loss": []}

        # loss function and optimizer
        loss_fn: torch.nn.L1Loss = torch.nn.L1Loss()  # mean square error
        optimizer: torch.optim.Adam = torch.optim.Adam(model.parameters())

        # number of epochs to run
        n_epochs: int = 10000

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
            print("Val mae: %.2f" % best_mae)
            if val_mae < best_mae:
                best_mae = val_mae
                # save the best model
                torch.save(model.state_dict(), mytools.get_model_path("pytorch_dnn_model.pt"))

            # clear cache
            # torch.cuda.empty_cache()

        # save history
        mytools.plot_history(_airport, history, f"pytorch_dnn_{_airport}_info.png")
        mytools.ModelRecords.update(_airport, "best_mae", best_mae, True)
