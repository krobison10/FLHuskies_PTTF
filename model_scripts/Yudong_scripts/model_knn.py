import mytools
import numpy as np
import pandas as pd  # type: ignore
from sklearn.neighbors import KNeighborsRegressor  # type: ignore
from sklearn.metrics import mean_absolute_error

_data_train: pd.DataFrame = mytools.get_train_tables().drop_duplicates(subset=["gufi"])
_data_test: pd.DataFrame = mytools.get_validation_tables().drop_duplicates(subset=["gufi"])

features: tuple[str, ...] = (
    "wind_direction",
    "wind_gust",
    "temperature",
    "delay_3hr",
    "delay_30hr",
    "standtime_3hr",
    "standtime_30hr",
)

X_train: np.ndarray = np.asarray([_data_train[_col] for _col in features], dtype="float32")
X_test: np.ndarray = np.asarray([_data_test[_col] for _col in features], dtype="float32")

X_train = np.reshape(X_train, (X_train.shape[1], X_train.shape[0]))
X_test = np.reshape(X_test, (X_test.shape[1], X_test.shape[0]))

y_train: np.ndarray = np.asarray(_data_train["minutes_until_pushback"])
y_test: np.ndarray = np.asarray(_data_test["minutes_until_pushback"])

model_knn = KNeighborsRegressor()

model_knn.fit(X_train, y_train)

result = model_knn.predict(X_test)

print(f"Validation mae: {mean_absolute_error(y_test, result)}\n")
