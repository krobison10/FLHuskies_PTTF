#
# Author: Yudong Lin
#
# A simple regression model implements with deep neural network
#

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint  # type: ignore

print(tf.__version__)


def plot_loss(history):
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.ylim([0, 10])
    plt.xlabel("Epoch")
    plt.ylabel("Error [MPG]")
    plt.legend()
    plt.grid(True)


def build_and_compile_model(norm):
    model = keras.Sequential([norm, layers.Dense(64, activation="relu"), layers.Dense(64, activation="relu"), layers.Dense(1)])

    model.compile(loss="mean_absolute_error", optimizer=tf.keras.optimizers.Adam(0.001))
    return model


_data_train = pd.DataFrame = pd.read_csv(
    os.path.join(os.path.dirname(__file__), "..", "train_tables", "KSEA_train.csv"),
    parse_dates=["timestamp"],
    dtype={"minutes_until_etd": int, "minutes_until_pushback": int},
)

_data_test = pd.DataFrame = pd.read_csv(
    os.path.join(os.path.dirname(__file__), "..", "validation_tables", "KSEA_validation.csv"),
    parse_dates=["timestamp"],
    dtype={"minutes_until_etd": int, "minutes_until_pushback": int},
)

X_train = np.asarray([_data_train["minutes_until_etd"]])
X_test = np.asarray([_data_test["minutes_until_etd"]])

X_train = np.reshape(X_train, (X_train.shape[1], X_train.shape[0]))
X_test = np.reshape(X_test, (X_test.shape[1], X_test.shape[0]))

y_train = np.asarray(_data_train["minutes_until_pushback"])
y_test = np.asarray(_data_test["minutes_until_pushback"])

horsepower_normalizer = layers.Normalization(input_shape=[X_train.shape[1]], axis=-1)
horsepower_normalizer.adapt(X_train)


model = Sequential([horsepower_normalizer])
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dense(512, activation="relu"))
model.add(layers.Dense(512, activation="relu"))
model.add(layers.Dense(1024, activation="relu"))
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(1))
model.compile(loss="mean_absolute_error", optimizer="adam")

model.summary()


# Model Checkpoint
check_pointer = ModelCheckpoint(
    os.path.join(os.path.dirname(__file__), "dnn_model.h5"),
    monitor="loss",
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    mode="auto",
    save_freq="epoch",
)
# Model Early Stopping Rules
early_stopping = EarlyStopping(monitor="val_loss", patience=5)

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), verbose=1, epochs=30, callbacks=[check_pointer, early_stopping])

print(history)

plot_loss(history)