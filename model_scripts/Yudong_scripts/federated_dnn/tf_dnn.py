#
# Author: Yudong Lin
#
# A simple regression model implements with deep neural network
#

import os

import mytools
import tensorflow as tf  # type: ignore
from constants import ALL_AIRPORTS, TARGET_LABEL
from sklearn.preprocessing import MinMaxScaler  # type: ignore

# allow gpu memory growth
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)


class MyTensorflowDNN:
    DEV_MODE: bool = False

    @classmethod
    def __get_model_path(cls, _airport: str) -> str:
        return mytools.get_model_path(f"tf_dnn_{_airport}_model")

    @classmethod
    def get_model(
        cls, _airport: str, _shape: tuple[int, ...], load_if_exists: bool = True
    ) -> tf.keras.models.Sequential:
        _model: tf.keras.models.Sequential
        model_path: str = cls.__get_model_path(_airport)
        if load_if_exists is False or not os.path.exists(model_path):
            print("----------------------------------------")
            print("Creating new model.")
            print("----------------------------------------")
            _layers: list[tf.keras.layers.Dense] = [
                tf.keras.layers.Input(_shape),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(1),
            ]
            _model = tf.keras.models.Sequential(_layers)
            _model.compile(loss="mean_absolute_error", optimizer=tf.keras.optimizers.Adam())
        else:
            print("----------------------------------------")
            print("A existing model has been found and will be loaded.")
            print("----------------------------------------")
            _model = tf.keras.models.load_model(model_path)

        return _model

    @classmethod
    def train(
        cls, _airport: str, using_a_normalizer: bool = True, load_if_exists: bool = True
    ) -> tf.keras.models.Sequential:
        normalizers: dict[str, MinMaxScaler] | None = mytools.get_normalizer() if using_a_normalizer is True else None

        # load train and test data frame
        train_df, val_df = mytools.get_train_and_test_ds(_airport, "PRIVATE_ALL")

        if normalizers is not None:
            for _col in train_df.columns:
                if _col != TARGET_LABEL and _col in normalizers:
                    train_df[[_col]] = normalizers[_col].transform(train_df[[_col]])
                    val_df[[_col]] = normalizers[_col].transform(val_df[[_col]])

        X_train: tf.Tensor = tf.convert_to_tensor(train_df.drop(columns=[TARGET_LABEL]), dtype=tf.float32 if normalizers is not None else None)
        X_test: tf.Tensor = tf.convert_to_tensor(val_df.drop(columns=[TARGET_LABEL]), dtype=tf.float32 if normalizers is not None else None)
        y_train: tf.Tensor = tf.convert_to_tensor(train_df[TARGET_LABEL], dtype=tf.int16)
        y_test: tf.Tensor = tf.convert_to_tensor(val_df[TARGET_LABEL], dtype=tf.int16)

        # load model
        model: tf.keras.models.Sequential = cls.get_model(_airport, (X_train.get_shape()[1],), load_if_exists)

        # show model info
        if cls.DEV_MODE is True:
            model.summary()

        # Model Checkpoint
        check_pointer: tf.keras.callbacks.ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint(
            cls.__get_model_path(_airport),
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode="auto",
            save_freq="epoch",
        )
        # Model Early Stopping Rules
        early_stopping: tf.keras.callbacks.EarlyStopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10
        )

        result = model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            verbose=1,
            epochs=50,
            callbacks=[check_pointer, early_stopping],
            batch_size=32 * 8,
        )

        # save history
        if cls.DEV_MODE is True:
            # show params
            print(result.params)
            # update database name
            mytools.ModelRecords.set_name("tf_dnn_model_records")
            # save loss history image
            mytools.plot_history(_airport, result.history, f"tf_dnn_{_airport}_info.png")
            # save loss history as json
            mytools.ModelRecords.update(_airport, "history", result.history, True)

        return model

    @classmethod
    def evaluate_global(cls) -> None:
        _model = tf.keras.models.load_model(cls.__get_model_path("ALL"))

        for theAirport in ALL_AIRPORTS:
            # load train and test data frame
            train_df, val_df = mytools.get_train_and_test_ds(theAirport)

            X_train: tf.Tensor = tf.convert_to_tensor(train_df.drop(columns=[TARGET_LABEL]))
            X_test: tf.Tensor = tf.convert_to_tensor(val_df.drop(columns=[TARGET_LABEL]))
            y_train: tf.Tensor = tf.convert_to_tensor(train_df[TARGET_LABEL], dtype=tf.int16)
            y_test: tf.Tensor = tf.convert_to_tensor(val_df[TARGET_LABEL], dtype=tf.int16)

            print(theAirport, ":")
            # _model.evaluate(X_train, y_train)
            _model.evaluate(X_test, y_test)
