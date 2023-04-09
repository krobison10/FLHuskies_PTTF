# @author:Daniil Filienko
from pathlib import Path
import pickle
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor, Dataset
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def plotImp(model, X, airport, num=20, fig_size=(40, 20)):
    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": X.columns})
    plt.figure(figsize=fig_size)
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title("LightGBM Features (avg over folds)")
    plt.tight_layout()
    plt.savefig(f"lgbm_importances_{airport}_local.png")

DATA_DIRECTORY = Path("./full_tables")
OUTPUT_DIRECTORY = Path("./models/Daniil_models")
AIRPORTS = ["KATL", "KCLT", "KDEN", "KDFW", "KJFK", "KMEM", "KMIA", "KORD", "KPHX", "KSEA"]
train = []

for airport in AIRPORTS:
    print("Started")
    train_airport = pd.read_csv(DATA_DIRECTORY / f"main_{airport}_prescreened.csv")
    train = train_airport.sort_values(by=["gufi"])

    train.rename(
        columns={
            "wind_direction": "wind_direction_cat",
            "cloud_ceiling": "cloud_ceiling_cat",
            "visibility": "visibility_cat",
        },
        inplace=True,
    )

    for c in train.columns:
        col_type = train[c].dtype
        if col_type == "object" or col_type == "string" or "cat" in c:
            train[c] = train[c].astype("category")

    print("Generated a shared dataframe")

    # Preventing GUFI from being an attribute to analyze
    offset = 4

    cat_features = [10, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    cat_features = [c - offset for c in cat_features]

    features_all = (train.columns.values.tolist())[offset : (len(train.columns.values))]

    features_remove = ()
    features = [x for x in features_all if x not in features_remove]
    # features_encoded = ["cloud_enc","aircraft_engine_class_enc","lightning_prob_enc","aircraft_type_enc","major_carrier_enc",
    #                                "flight_type_enc","gufi_end_label_enc","wind_direction_enc","precip_enc"]
    # features = features_all + features_encoded
    X_train = Dataset(data=train[features])
    y_train = Dataset(data=train[features])

    print(features)

    X_train = train[features]

    y_train = train["minutes_until_pushback"]

    # Remove the testing of the features
    # X_test = test[features]

    # y_test = test["minutes_until_pushback"]
    fit_params = {
        "eval_metric": "MAE",
        "verbose": 100,
        "feature_name": "auto",  # that's actually the default
        "categorical_feature": "auto",  # that's actually the default
    }
    ensembleRegressor = LGBMRegressor(objective="regression_l1")
    # ensembleRegressor.fit(X_train, y_train,cat_features=cat_features,use_best_model=True)

    ensembleRegressor.fit(X_train, y_train, **fit_params)

    print("Finished training")

    # y_pred = test["minutes_until_departure"] - 15
    # print("Baseline:", mean_absolute_error(y_test, y_pred))

    # y_pred = regressor.predict(X_train)
    # print("Regression tree train error:", mean_absolute_error(y_train, y_pred))

    # Remove the evaluation of the model
    # y_pred = ensembleRegressor.predict(X_test)
    # print("Ensemble of tree regressors test error:", mean_absolute_error(y_test, y_pred))
    filename = f"model_w_mfs_lamp_time_etd_{airport}_lightgmb.sav"
    pickle.dump(ensembleRegressor, open(OUTPUT_DIRECTORY / filename, "wb"))
    print("Saved the model for the airport: ", airport)
