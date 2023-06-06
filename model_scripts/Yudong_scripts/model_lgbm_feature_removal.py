from typing import Any

import lightgbm as lgb  # type: ignore
import mytools
import pandas as pd
from constants import TARGET_LABEL
from sklearn.metrics import mean_absolute_error  # type: ignore
from sklearn.preprocessing import OrdinalEncoder  # type: ignore

ignore_features = [
    "gufi",
    "timestamp",
    "gufi_flight_date",
    "gufi_flight_number",
    "isdeparture",
    "dep_ratio",
    "arr_ratio",
]

encoded_columns: list[str] = [
    "cloud",
    "lightning_prob",
    "precip",
    "aircraft_engine_class",
    "aircraft_type",
    "major_carrier",
    "flight_type",
    "departure_runways",
    "arrival_runways",
    "gufi_flight_destination_airport",
    "gufi_flight_FAA_system",
    "airline",
]

for _ignore in ignore_features:
    if _ignore in encoded_columns:
        encoded_columns.remove(_ignore)

airport = "KMEM"

train_df: pd.DataFrame = mytools.get_train_tables(airport, remove_duplicate_gufi=False)
train_df.drop(columns=ignore_features, inplace=True)

val_df: pd.DataFrame = mytools.get_validation_tables(airport, remove_duplicate_gufi=False)
val_df.drop(columns=ignore_features, inplace=True)

for col in encoded_columns:
    encoder: OrdinalEncoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    encoded_col = encoder.fit_transform(train_df[[col]])
    train_df[[col]] = encoded_col

    encoded_col = encoder.transform(val_df[[col]])
    val_df[[col]] = encoded_col

    train_df[col] = train_df[col].astype(int)
    val_df[col] = val_df[col].astype(int)

# Define the datasets with the categorical_feature parameter
X_train = train_df.drop(columns=[TARGET_LABEL])
X_test = val_df.drop(columns=[TARGET_LABEL])

y_train = train_df[TARGET_LABEL]
y_test = val_df[TARGET_LABEL]

params: dict[str, Any] = {
    "boosting_type": "gbdt",
    "objective": "regression_l1",
    "num_leaves": 1024 * 8,
    "num_iterations": 128,
}

model = lgb.train(
    params, lgb.Dataset(X_train, y_train), lgb.Dataset(X_test, y_test), categorical_feature=encoded_columns
)

mytools.log_importance(model)

y_pred = model.predict(X_train)
mae: float = round(mean_absolute_error(y_train, y_pred), 4)
print(f"MAE on train data {airport}: {mae}")
y_pred = model.predict(X_test)
mae = round(mean_absolute_error(y_test, y_pred), 4)
print(f"MAE on validation data {airport}: {mae}")
