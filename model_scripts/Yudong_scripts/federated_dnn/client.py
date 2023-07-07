import flwr
import mytools
import tensorflow as tf  # type: ignore
from constants import TARGET_LABEL
from tf_dnn import MyTensorflowDNN


class CifarClient(flwr.client.NumPyClient):
    def __init__(self, _airport: str, _airline: str) -> None:
        super().__init__()

        train_df, val_df = mytools.get_train_and_test_ds(_airport, _airline)

        self.__X_train: tf.Tensor = tf.convert_to_tensor(train_df.drop(columns=[TARGET_LABEL]))
        self.__X_test: tf.Tensor = tf.convert_to_tensor(val_df.drop(columns=[TARGET_LABEL]))

        self.__y_train: tf.Tensor = tf.convert_to_tensor(train_df[TARGET_LABEL])
        self.__y_test: tf.Tensor = tf.convert_to_tensor(val_df[TARGET_LABEL])

        self.__model = MyTensorflowDNN.get_model(_airport, (self.__X_train.get_shape()[1],), load_if_exists=False)

    def get_parameters(self, config):
        return self.__model.get_weights()

    def fit(self, parameters, config):
        self.__model.set_weights(parameters)
        self.__model.fit(self.__X_train, self.__y_train, epochs=1, batch_size=32, steps_per_epoch=3)
        return self.__model.get_weights(), len(self.__X_train), {}

    def evaluate(self, parameters, config):
        self.__model.set_weights(parameters)
        loss = self.__model.evaluate(self.__X_test, self.__y_test)
        return loss, len(self.__X_test), {"loss": loss}


if __name__ == "__main__":
    import argparse

    import flwr

    # using argparse to parse the argument from command line
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("-p", help="airport")
    parser.add_argument("-l", help="airline")
    parser.add_argument("-i", help="ip")
    args: argparse.Namespace = parser.parse_args()

    flwr.client.start_numpy_client(
        server_address=str(args.i).lower() if args.i is not None else "[::]:8080",
        client=CifarClient(
            str(args.p).upper() if args.p is not None else "ALL",
            str(args.l).upper() if args.l is not None else "PRIVATE_ALL",
        ),
    )
