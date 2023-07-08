#
# Author: Yudong Lin
#
# A simple regression model implements with deep neural network
#

import os

import mytools
import numpy
import tensorflow as tf  # type: ignore
from constants import ALL_AIRPORTS, TARGET_LABEL
from sklearn.metrics import mean_absolute_error

_features_using: list[str] = [
    "minutes_until_etd",
    "deps_3hr",
    "airport",
    "deps_30hr",
    "arrs_3hr",
    "arrs_30hr",
    "deps_taxiing",
    "exp_deps_15min",
    "exp_deps_30min",
    "delay_30hr",
    "standtime_30hr",
    "dep_taxi_30hr",
    "delay_3hr",
    "standtime_3hr",
    "dep_taxi_3hr",
    "1h_ETDP",
    "departure_runways",
    "arrival_runways",
    "temperature",
    "wind_direction",
    "wind_speed",
    "wind_gust",
    "cloud_ceiling",
    "cloud",
    "lightning_prob",
    "airline",
    "gufi_flight_destination_airport",
    "gufi_timestamp_until_etd",
    "year",
    "month",
    "day",
    "hour",
    "minute",
    "weekday",
    "aircraft_type",
    "major_carrier",
    "visibility",
    "flight_type",
    TARGET_LABEL,
]

# allow gpu memory growth
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)


class MyTensorflowDNN:
    @classmethod
    def __get_model_path(cls, _airport: str) -> str:
        return mytools.get_model_path(f"tf_dnn_{_airport}_model")

    @classmethod
    def get_model(
        cls, _airport: str, _normalizer: tf.keras.layers.Normalization, load_if_exists: bool = True
    ) -> tf.keras.models.Sequential:
        _model: tf.keras.models.Sequential
        model_path: str = cls.__get_model_path(_airport)
        if load_if_exists is False or not os.path.exists(model_path):
            print("----------------------------------------")
            print("Creating new model.")
            print("----------------------------------------")
            _model = tf.keras.models.Sequential(
                [
                    _normalizer,
                    tf.keras.layers.Dense(32, activation="relu"),
                    tf.keras.layers.Dense(64, activation="relu"),
                    tf.keras.layers.Dense(64, activation="relu"),
                    tf.keras.layers.Dense(1),
                ]
            )
            _model.compile(loss="mean_absolute_error", optimizer=tf.keras.optimizers.Adam())
        else:
            print("----------------------------------------")
            print("A existing model has been found and will be loaded.")
            print("----------------------------------------")
            _model = tf.keras.models.load_model(model_path)

        return _model

    @classmethod
    def train(cls, _airport: str) -> None:
        # update database name
        mytools.ModelRecords.set_name("tf_dnn_model_records")

        # load train and test data frame
        train_df, val_df = mytools.get_train_and_test_ds(_airport, use_cols=_features_using)

        X_train: tf.Tensor = tf.convert_to_tensor(train_df.drop(columns=[TARGET_LABEL]))
        X_test: tf.Tensor = tf.convert_to_tensor(val_df.drop(columns=[TARGET_LABEL]))
        y_train: tf.Tensor = tf.convert_to_tensor(train_df[TARGET_LABEL], dtype=tf.int16)
        y_test: tf.Tensor = tf.convert_to_tensor(val_df[TARGET_LABEL], dtype=tf.int16)

        normalizer: tf.keras.layers.Normalization = tf.keras.layers.Normalization(axis=-1)
        normalizer.adapt(X_test)
        normalizer.adapt(X_train)

        # load model
        model: tf.keras.models.Sequential = cls.get_model(_airport, normalizer)

        model.summary()

        print(train_df.columns)

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

        print(result.params)

        # save history
        mytools.plot_history(_airport, result.history, f"tf_dnn_{_airport}_info.png")
        mytools.ModelRecords.update(_airport, "history", result.history, True)

    @classmethod
    def evaluate(cls, theAirport: str) -> None:
        if theAirport == "ALL":
            cls.evaluate_global()
            return

        _model = tf.keras.models.load_model(cls.__get_model_path(theAirport))

        # load train and test data frame
        train_df, val_df = mytools.get_train_and_test_ds(theAirport, use_cols=_features_using)

        # X_train: tf.Tensor = tf.convert_to_tensor(train_df.drop(columns=[TARGET_LABEL]))
        X_test: tf.Tensor = tf.convert_to_tensor(val_df.drop(columns=[TARGET_LABEL]))
        # y_train: tf.Tensor = tf.convert_to_tensor(train_df[TARGET_LABEL], dtype=tf.int16)
        y_test: tf.Tensor = tf.convert_to_tensor(val_df[TARGET_LABEL], dtype=tf.int16)

        print(theAirport, ":")
        # _model.evaluate(X_train, y_train)
        _model.evaluate(X_test, y_test)

    @classmethod
    def evaluate_global(cls) -> None:
        _model = tf.keras.models.load_model(cls.__get_model_path("ALL"))

        for theAirport in ALL_AIRPORTS:
            # load train and test data frame
            train_df, val_df = mytools.get_train_and_test_ds(theAirport, use_cols=_features_using)

            # X_train: tf.Tensor = tf.convert_to_tensor(train_df.drop(columns=[TARGET_LABEL]))
            X_test: tf.Tensor = tf.convert_to_tensor(val_df.drop(columns=[TARGET_LABEL]))
            # y_train: tf.Tensor = tf.convert_to_tensor(train_df[TARGET_LABEL], dtype=tf.int16)
            y_test: tf.Tensor = tf.convert_to_tensor(val_df[TARGET_LABEL], dtype=tf.int16)

            print(theAirport, ":")
            # _model.evaluate(X_train, y_train)
            _model.evaluate(X_test, y_test)

    @classmethod
    def evaluate_individual_in_global_setting(cls) -> None:
        y_real = numpy.asarray([], dtype=numpy.int32)
        y_pred = numpy.asarray([], dtype=numpy.int32)
        for theAirport in ALL_AIRPORTS:
            _model = tf.keras.models.load_model(cls.__get_model_path(theAirport))

            # load train and test data frame
            train_df, val_df = mytools.get_train_and_test_ds(theAirport, use_cols=_features_using)

            # X_train: tf.Tensor = tf.convert_to_tensor(train_df.drop(columns=[TARGET_LABEL]))
            X_test: tf.Tensor = tf.convert_to_tensor(val_df.drop(columns=[TARGET_LABEL]))

            y_pred = numpy.concatenate((y_pred, _model.predict(X_test).ravel()))
            y_real = numpy.concatenate((y_real, val_df[TARGET_LABEL].to_numpy()))

        print("Result:", mean_absolute_error(y_real, y_pred))
