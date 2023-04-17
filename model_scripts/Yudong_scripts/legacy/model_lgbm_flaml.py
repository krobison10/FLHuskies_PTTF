from datetime import datetime

# import lightgbm  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import mytools
import pandas as pd  # type: ignore
from flaml import AutoML  # type: ignore
from flaml.default import LGBMRegressor  # type: ignore
from joblib import dump  # type: ignore
from sklearn.metrics import mean_absolute_error  # type: ignore
from sklearn.preprocessing import OrdinalEncoder  # type: ignore

TARGET_LABEL: str = "minutes_until_pushback"

ignore_features = [
    "gufi",
    "timestamp",
    "airport",
    "gufi_flight_date",
    "isdeparture",
    "dep_ratio",
    "arr_ratio",
    "gufi_flight_number",
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
    "gufi_flight_major_carrier",
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

automl_settings = {
    "metric": "mae",
    "task": "regression",
}

model = AutoML()
model.fit(X_train, y_train, **automl_settings)

y_pred = model.predict(X_train)
mae: float = round(mean_absolute_error(y_train, y_pred), 4)
print(f"MAE on train data {airport}: {mae}")
y_pred = model.predict(X_test)
mae = round(mean_absolute_error(y_test, y_pred), 4)
print(f"MAE on validation data {airport}: {mae}")

# plot the graph that shows importance
model_name: str = datetime.now().strftime("%Y%m%d_%H%M%S")

# plot the graph that shows importance
# lightgbm.plot_importance(model, ignore_zero=False)
# plt.savefig(mytools.get_model_path(f"lgbm_{airport}_{model_name}_importance.png"), bbox_inches="tight")

dump(model, mytools.get_model_path(f"lgbm_{airport}_model.joblib"))
