#
# Author: Yudong Lin
#
# A simple regression model implements with deep neural network
#

import os
import mytools
import tensorflow as tf  # type: ignore
from constants import TARGET_LABEL


class MyDNN:
    @staticmethod
    def __get_model_path(_airport: str) -> str:
        return mytools.get_model_path(f"tf_dnn_{_airport}_model.h5")

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
                    tf.keras.layers.Dense(64, activation="elu"),
                    tf.keras.layers.Dense(64, activation="elu"),
                    tf.keras.layers.Dense(64, activation="elu"),
                    tf.keras.layers.Dense(64, activation="elu"),
                    tf.keras.layers.Dense(128, activation="elu"),
                    tf.keras.layers.Dense(128, activation="elu"),
                    tf.keras.layers.Dense(128, activation="elu"),
                    tf.keras.layers.Dense(128, activation="elu"),
                    tf.keras.layers.Dense(256, activation="elu"),
                    tf.keras.layers.Dense(256, activation="elu"),
                    tf.keras.layers.Dense(256, activation="elu"),
                    tf.keras.layers.Dense(256, activation="elu"),
                    tf.keras.layers.Dense(512, activation="elu"),
                    tf.keras.layers.Dense(512, activation="elu"),
                    tf.keras.layers.Dense(1024, activation="elu"),
                    tf.keras.layers.Dense(1024, activation="elu"),
                    tf.keras.layers.Dense(1),
                ]
            )
            _model.compile(loss="mean_absolute_error", optimizer="adam")
        else:
            print("----------------------------------------")
            print("A existing model has been found and will be loaded.")
            print("----------------------------------------")
            _model = tf.keras.models.load_model(model_path)

        return _model

    @classmethod
    def train_dnn(cls, _airport: str) -> None:
        # load train and test data frame
        train_df, val_df = mytools.get_train_and_test_ds(_airport)

        X_train: tf.Tensor = tf.convert_to_tensor(train_df.drop(columns=[TARGET_LABEL]))
        X_test: tf.Tensor = tf.convert_to_tensor(val_df.drop(columns=[TARGET_LABEL]))

        normalizer: tf.keras.layers.Normalization = tf.keras.layers.Normalization(axis=-1)
        normalizer.adapt(X_test)
        normalizer.adapt(X_train)

        y_train: tf.Tensor = tf.convert_to_tensor(train_df[TARGET_LABEL])
        y_test: tf.Tensor = tf.convert_to_tensor(val_df[TARGET_LABEL])

        model: tf.keras.models.Sequential = cls.get_model(_airport, normalizer)

        model.summary()

        # Model Checkpoint
        check_pointer: tf.keras.callbacks.ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint(
            cls.__get_model_path(_airport),
            monitor="loss",
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

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            verbose=1,
            epochs=30,
            callbacks=[check_pointer, early_stopping],
        )

        print(history.params)

        mytools.plot_loss(history)


if __name__ == "__main__":
    MyDNN.train_dnn("KSEA")
