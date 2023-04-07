#
# Daniil Filienko
#
# Running CatBoostRegressor with Kyler's Train Split for Trevor
# to attain accuracy values for individual airports and overall

import matplotlib.pyplot as plt
from table_scripts.train_test_split import *
import pandas as pd
import pickle
import numpy as np
from pathlib import Path
import seaborn as sns
from lightgbm import LGBMRegressor, Dataset
from sklearn.preprocessing import OrdinalEncoder

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
    plt.savefig(f"lgbm_importances_{airport}_trevor.png")


y_tests = [0]
y_preds = [0]
for airport in airports:
    # replace this path with the locations of the full tables for each airport if necessary
    df = pd.read_csv(os.path.join(ROOT, DATA_DIRECTORY, f"main_{airport}_prescreened.csv"))

    train_df, val_df = split(table=df, airport=airport, save=False)

    train_df["precip"] = train_df["precip"].astype(str)
    val_df["precip"] = val_df["precip"].astype(str)

    enc0 = OrdinalEncoder()
    train_df["lightning_prob_enc"] = enc0.fit_transform(train_df[["lightning_prob"]].values)
    val_df["lightning_prob_enc"] = enc0.transform(val_df[["lightning_prob"]].values)

    enc1 = OrdinalEncoder()
    train_df["cloud_enc"] = enc1.fit_transform(train_df[["cloud"]].values)
    val_df["cloud_enc"] = enc1.transform(val_df[["cloud"]].values)

    enc2 = OrdinalEncoder()
    train_df["aircraft_engine_class_enc"] = enc2.fit_transform(train_df[["aircraft_engine_class"]].values)
    val_df["aircraft_engine_class_enc"] = enc2.transform(val_df[["aircraft_engine_class"]].values)

    enc3 = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    train_df["aircraft_type_enc"] = enc3.fit_transform(train_df[["aircraft_type"]].values)
    val_df["aircraft_type_enc"] = enc3.transform(val_df[["aircraft_type"]].values)

    enc4 = OrdinalEncoder()
    train_df["major_carrier_enc"] = enc4.fit_transform(train_df[["major_carrier"]].values)
    val_df["major_carrier_enc"] = enc4.transform(val_df[["major_carrier"]].values)

    enc5 = OrdinalEncoder()
    train_df["flight_type_enc"] = enc5.fit_transform(train_df[["flight_type"]].values)
    val_df["flight_type_enc"] = enc5.transform(val_df[["flight_type"]].values)

    enc6 = OrdinalEncoder()
    train_df["precip_enc"] = enc6.fit_transform(train_df[["precip"]].values)
    val_df["precip_enc"] = enc6.transform(val_df[["precip"]].values)

    features = [
        "minutes_until_etd",
        "aircraft_engine_class_enc",
        "aircraft_type_enc",
        "major_carrier_enc",
        "flight_type_enc",
        "temperature",
        "wind_direction",
        "wind_speed",
        "wind_gust",
        "cloud_ceiling",
        "visibility",
        "cloud_enc",
        "lightning_prob_enc",
        "precip_enc",
    ]

    # ---------------------------------------- BASELINE ----------------------------------------
    # add columns representing standard and improved baselines to validation table
    val_df["baseline"] = val_df.apply(lambda row: max(row["minutes_until_etd"] - 15, 0), axis=1)
    # print performance of baseline estimates
    mae = mean_absolute_error(val_df["minutes_until_pushback"], val_df["baseline"])
    print(f"\nMAE for {airport} with baseline: {mae:.4f}")

    # evaluating individual airport accuracy
    print(f"Training LIGHTGBM model for {airport}\n")
    X_train = (train_df[features]).to_numpy()
    X_test = (val_df[features]).to_numpy()
    y_train = (train_df["minutes_until_pushback"]).to_numpy()
    y_test = (val_df["minutes_until_pushback"]).to_numpy()

    fit_params = {
        "eval_metric": "MAE",
        "verbose": 100,
        "feature_name": "auto",  # that's actually the default
        "categorical_feature": "auto",  # that's actually the default
    }

    gbm = LGBMRegressor(objective="regression_l1")
    gbm.fit(X_train, y_train, eval_metric="l1")

    # ensembleRegressor.fit(X_train, y_train,cat_features=cat_features,use_best_model=True)

    # ensembleRegressor.fit(X_train, y_train, **fit_params)
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)

    print("Finished training")
    print(f"MAE on {airport} test data: {mean_absolute_error(y_test, y_pred):.4f}\n")
    # appending the predictions and test to a single datasets to evaluate overall performance
    y_tests = np.concatenate((y_tests, y_test))
    y_preds = np.concatenate((y_preds, y_pred))

    # plotImp(gbm,X_test,airport=airport)

    # y_tests.append(y_test)
    # y_preds.append(y_pred)

# y_tests = np.hstack(y_tests)
# y_pred = np.hstack(y_preds)
print(f"MAE on all test data: {mean_absolute_error(y_tests, y_preds):.4f}\n")

exit()
