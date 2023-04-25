#
# Author: Yudong Lin
#
# A simple regression model implements with deep neural network
#

import mytools
import numpy as np
import tensorflow as tf  # type: ignore
from tensorflow.keras import layers  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore

print(tf.__version__)


def train_dnn(_airport: str) -> None:
    _data_train, _data_test = mytools.get_train_and_test_ds(_airport)

    model = mytools.get_model(_airport)

    _data_train["lgbm_prediction"] = model.predict(_data_train.drop(columns=["minutes_until_pushback"]))
    _data_test["lgbm_prediction"] = model.predict(_data_test.drop(columns=["minutes_until_pushback"]))

    features: tuple[str, ...] = ("lgbm_prediction",)

    X_train: np.ndarray = np.asarray([_data_train[_col] for _col in features], dtype="float32")
    X_test: np.ndarray = np.asarray([_data_test[_col] for _col in features], dtype="float32")

    X_train = np.reshape(X_train, (X_train.shape[1], X_train.shape[0]))
    X_test = np.reshape(X_test, (X_test.shape[1], X_test.shape[0]))

    y_train: np.ndarray = np.asarray(_data_train["minutes_until_pushback"])
    y_test: np.ndarray = np.asarray(_data_test["minutes_until_pushback"])

    model = Sequential()
    model.add(layers.Dense(64, activation="elu", input_shape=(X_train.shape[1],)))
    model.add(layers.Dense(64, activation="elu"))
    model.add(layers.Dense(128, activation="elu"))
    model.add(layers.Dense(128, activation="elu"))
    model.add(layers.Dense(256, activation="elu"))
    model.add(layers.Dense(256, activation="elu"))
    model.add(layers.Dense(512, activation="elu"))
    model.add(layers.Dense(512, activation="elu"))
    model.add(layers.Dense(1024, activation="elu"))
    model.add(layers.Dense(1024, activation="elu"))
    model.add(layers.Dense(1))
    model.compile(loss="mean_absolute_error", optimizer="adam")

    model.summary()

    # Model Checkpoint
    check_pointer = ModelCheckpoint(
        mytools.get_model_path("dnn_model.h5"),
        monitor="loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode="auto",
        save_freq="epoch",
    )
    # Model Early Stopping Rules
    early_stopping = EarlyStopping(monitor="val_loss", patience=10)

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        verbose=1,
        epochs=30,
        callbacks=[check_pointer, early_stopping],
    )

    print(history)

    mytools.plot_loss(history)


if __name__ == "__main__":
    train_dnn("KSEA")
