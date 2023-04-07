#
# Daniil Filienko
#
# Running CatBoostRegressor with Kyler's Train Split.
# to attain accuracy values for individual airports and overall

import matplotlib.pyplot as plt
from table_scripts.train_test_split import *
import pandas as pd
import pickle
import numpy as np
from pathlib import Path
import seaborn as sns

from sklearn.metrics import mean_absolute_error

# ---------------------------------------- MAIN ----------------------------------------
DATA_DIRECTORY = Path("full_tables")

airports = [
    "KATL",
    "KCLT",
    "KDEN",
    "KDFW",
    "KJFK",
    "KMEM",
    "KMIA",
    "KORD",
    "KPHX",
    "KSEA",
]


def plotImp(model, X, airport, num=20, fig_size=(40, 20)):
    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": X.columns})
    plt.figure(figsize=fig_size)
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title("LightGBM Features (avg over folds)")
    plt.tight_layout()
    plt.savefig(f"lgbm_importances_{airport}.png")


y_tests = [0]
y_preds = [0]
for airport in airports:
    # replace this path with the locations of the full tables for each airport if necessary
    df = pd.read_csv(os.path.join(ROOT, DATA_DIRECTORY, f"main_{airport}_prescreened.csv"))
    train_df, val_df = split(table=df, airport=airport, save=False)

    val_df.rename(
        columns={
            "wind_direction": "wind_direction_cat",
            "cloud_ceiling": "cloud_ceiling_cat",
            "visibility": "visibility_cat",
        },
        inplace=True,
    )

    # Lighgbm specific implementation
    for c in val_df.columns:
        col_type = val_df[c].dtype
        if col_type == "object" or col_type == "string" or "cat" in c:
            val_df[c] = val_df[c].astype("category")

    print("Finished the split")
    offset = 4
    features = (val_df.columns.values.tolist())[offset : (len(val_df.columns.values))]

    # ---------------------------------------- BASELINE ----------------------------------------
    # add columns representing standard and improved baselines to validation table
    val_df["baseline"] = val_df.apply(lambda row: max(row["minutes_until_etd"] - 15, 0), axis=1)
    # print performance of baseline estimates
    mae = mean_absolute_error(val_df["minutes_until_pushback"], val_df["baseline"])
    print(f"\nMAE for {airport} with baseline: {mae:.4f}")

    # evaluating individual airport accuracy
    print(f"Loading CatBoostRegressor Regressor for {airport}\n")
    model = pickle.load(open(f"./models/Daniil_models/model_w_mfs_lamp_time_etd_{airport}_lightgmb.sav", "rb"))
    X_test = val_df[features]
    y_test = val_df["minutes_until_pushback"]
    y_pred = model.predict(X_test)
    print(f"MAE on {airport} test data: {mean_absolute_error(y_test, y_pred):.4f}\n")
    # appending the predictions and test to a single datasets to evaluate overall performance
    y_tests = np.concatenate((y_tests, y_test))
    y_preds = np.concatenate((y_preds, y_pred))
    plotImp(model, X_test, airport=airport)

print(f"MAE on all test data: {mean_absolute_error(y_tests, y_preds):.4f}\n")

exit()
