#
# Daniil Filienko
#
# Running CatBoostRegressor with Kyler's Train Split.

from train_test_split import *
import pandas as pd
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import OrdinalEncoder
import pickle
import numpy as np

from sklearn.metrics import mean_absolute_error

# ---------------------------------------- MAIN ----------------------------------------
airports = [
    # "KATL",
    # "KCLT",
    # "KDEN",
    # "KDFW",
    # "KJFK",
    "KMEM",
    # "KMIA",
    # "KORD",
    # "KPHX",
    # "KSEA",
]
y_tests = [0]
y_preds = [0]
for airport in airports:
    # replace this path with the locations of the full tables for each airport if necessary
    df = pd.read_csv(os.path.join(ROOT, "full_tables", f"main_{airport}_prescreened.csv"))
    train_df, val_df = split(table=df, airport=airport, save=False)
    print("Finished the split")
    offset = 4
    features = (val_df.columns.values.tolist())[offset:(len(val_df.columns.values))]

    # ---------------------------------------- BASELINE ----------------------------------------
    # add columns representing standard and improved baselines to validation table
    val_df["baseline"] = val_df.apply(lambda row: max(row["minutes_until_etd"] - 15, 0), axis=1)
    # print performance of baseline estimates
    mae = mean_absolute_error(val_df["minutes_until_pushback"], val_df["baseline"])
    print(f"\nMAE for {airport} with baseline: {mae:.4f}")

    # evaluating individual airport accuracy
    print(f"Loading CatBoostRegressor Regressor for {airport}\n")
    model = pickle.load(open(f'./models/Daniil_models/model_w_mfs_lamp_time_etd_{airport}.sav', 'rb'))
    X_test = val_df[features]
    y_test = val_df["minutes_until_pushback"]
    y_pred = model.predict(X_test)
    print(f"MAE on {airport} test data: {mean_absolute_error(y_test, y_pred):.4f}\n")

    # appending the predictions and test to a single datasets to evaluate overall performance
    y_tests = np.concatenate((y_tests, y_test))
    y_preds = np.concatenate((y_preds, y_pred))

    # y_tests.append(y_test)
    # y_preds.append(y_pred)

# y_tests = np.hstack(y_tests)
# y_pred = np.hstack(y_preds)
print(f"MAE on all test data: {mean_absolute_error(y_tests, y_preds):.4f}\n")

exit()