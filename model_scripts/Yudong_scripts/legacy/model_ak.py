import autokeras as ak  # type: ignore
import mytools
import numpy as np
import pandas as pd  # type: ignore

_data_train: pd.DataFrame = mytools.get_train_tables()
_data_test: pd.DataFrame = mytools.get_validation_tables()

features: dict[str, tuple[str, ...]] = {
    "lamp": (
        "temperature",
        "wind_direction",
        "wind_speed",
        "wind_gust",
        "cloud_ceiling",
        "visibility",
        "cloud",
        "lightning_prob",
        "precip",
    ),
    "mfs": ("aircraft_engine_class", "aircraft_type", "major_carrier", "flight_type"),
    "time": ("standtime_3hr", "standtime_30hr", "month", "day", "hour", "weekday"),
}

mytools.encodeStrFeatures(_data_train, _data_test, *features["mfs"])
mytools.encodeStrFeatures(_data_train, _data_test, "cloud", "lightning_prob", "precip")

features_in_use: tuple[str, ...] = ("minutes_until_etd",)

X_train: np.ndarray = np.asarray([_data_train[_col] for _col in features_in_use], dtype="float32")
X_test: np.ndarray = np.asarray([_data_test[_col] for _col in features_in_use], dtype="float32")

X_train = np.reshape(X_train, (X_train.shape[1], X_train.shape[0]))
X_test = np.reshape(X_test, (X_test.shape[1], X_test.shape[0]))

y_train: np.ndarray = np.asarray(_data_train["minutes_until_pushback"])
y_test: np.ndarray = np.asarray(_data_test["minutes_until_pushback"])


input_node = ak.StructuredDataInput()
output_node = ak.StructuredDataBlock(categorical_encoding=True)(input_node)
output_node = ak.RegressionHead()(output_node)

# Initialize the structured data regressor.
reg = ak.AutoModel(inputs=input_node, outputs=output_node, overwrite=True, max_trials=50)

# Feed the structured data regressor with training data.
reg.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
