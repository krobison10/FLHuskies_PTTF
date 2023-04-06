# M. De Cock; Mar 24, 2023
from pathlib import Path
import catboost as cb
import pickle
from sklearn.metrics import mean_absolute_error

import numpy as np
import pandas as pd
DATA_DIRECTORY = Path("./train_tables")
OUTPUT_DIRECTORY = Path("./models/Daniil_models")
AIRPORTS = [
    "KATL",
    "KCLT",
    "KDEN",
    "KDFW",
    "KJFK",
    "KMEM",
    "KMIA",
    "KORD",
    "KPHX",
    "KSEA"
]
train = []

for airport in AIRPORTS:
    train_airport = pd.read_csv(DATA_DIRECTORY / f"main_{airport}_prescreened.csv")
    train_airport = train_airport.sort_values(by=['gufi'])
    # #For the combined model training, comment out following 2 lines, comment in following line
    # #and remove the intend for the following sections of training code
    # train.append(train_airport)
# train = pd.concat(train)

    train = train_airport
    #Split into train and test datasets
    # test = train.iloc[round(train.shape[0]*0.99):]
    # train = train.iloc[:round(train.shape[0]*0.1)]

    # make sure that the categorical features are encoded as strings
    cat_feature = train.columns[np.where(train.dtypes != float)[0]].values.tolist()
    train[cat_feature] = train[cat_feature].astype(str)

    #remove test for training the models
    # test[cat_feature] = test[cat_feature].astype(str)

    print("Generated a shared dataframe")
    # enc01 = OrdinalEncoder()
    # train["cloud_ceiling_enc"] = enc0.fit_transform(train[["cloud_ceiling"]].values)
    # test["cloud_ceiling_enc"] = enc0.transform(test[["cloud_ceiling"]].values)

    # enc0 = OrdinalEncoder()
    # train["lightning_prob_enc"] = enc0.fit_transform(train[["lightning_prob"]].values)
    # test["lightning_prob_enc"] = enc0.transform(test[["lightning_prob"]].values)

    # enc1 = OrdinalEncoder()
    # train["cloud_enc"] = enc1.fit_transform(train[["cloud"]].values)
    # test["cloud_enc"] = enc1.transform(test[["cloud"]].values)

    # enc2 = OrdinalEncoder()
    # train["aircraft_engine_class_enc"] = enc2.fit_transform(train[["aircraft_engine_class"]].values)
    # test["aircraft_engine_class_enc"] = enc2.transform(test[["aircraft_engine_class"]].values)

    # enc3 = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value = -1)
    # train["aircraft_type_enc"] = enc3.fit_transform(train[["aircraft_type"]].values)
    # test["aircraft_type_enc"] = enc3.transform(test[["aircraft_type"]].values)

    # enc4 = OrdinalEncoder()
    # train["major_carrier_enc"] = enc4.fit_transform(train[["major_carrier"]].values)
    # test["major_carrier_enc"] = enc4.transform(test[["major_carrier"]].values)

    # enc5 = OrdinalEncoder()
    # train["flight_type_enc"] = enc5.fit_transform(train[["flight_type"]].values)
    # test["flight_type_enc"] = enc5.transform(test[["flight_type"]].values)

    # enc6 = OrdinalEncoder()
    # train["gufi_end_label_enc"] = enc6.fit_transform(train[["gufi_end_label"]].values)
    # test["gufi_end_label_enc"] = enc6.transform(test[["gufi_end_label"]].values)

    # enc7 = OrdinalEncoder()
    # train["wind_direction_enc"] = enc7.fit_transform(train[["wind_direction"]].values)
    # test["wind_direction_enc"] = enc7.transform(test[["wind_direction"]].values)

    # enc8 = OrdinalEncoder()
    # train["precip_enc"] = enc8.fit_transform(train[["precip"]].values)
    # test["precip_enc"] = enc8.transform(test[["precip"]].values)

    # Preventing GUFI from being an attribute to analyze
    offset = 4

    cat_features = [15,16,17,18,19,20,21,22,23]
    cat_features = [c - offset for c in cat_features]

    # #For mfs only
    # cat_features = [5,6,7,8,9]

    # features_all = (train.columns.values.tolist())[offset:15]
    features_all = (train.columns.values.tolist())[offset:(len(train.columns.values))]
    # features_all = (train.columns.values.tolist())[offset:5]

    #For mfs only
    # features_remove = ("departure_runway_actual","cloud","aircraft_engine_class","lightning_prob","aircraft_type","major_carrier",
    #                                 "flight_type","gufi_end_label","precip")
    features_remove = ()
    features = [x for x in features_all if x not in features_remove]
    # features_encoded = ["cloud_enc","aircraft_engine_class_enc","lightning_prob_enc","aircraft_type_enc","major_carrier_enc",
    #                                "flight_type_enc","gufi_end_label_enc","wind_direction_enc","precip_enc"]
    # features = features_all + features_encoded

    print(features)

    X_train = train[features]

    y_train = train["minutes_until_pushback"]

    # Remove the testing of the features
    # X_test = test[features]

    # y_test = test["minutes_until_pushback"]

    ensembleRegressor = cb.CatBoostRegressor(has_time=True, loss_function="MAE", task_type="GPU", n_estimators=8000)
    # ensembleRegressor.fit(X_train, y_train,cat_features=cat_feature,use_best_model=True)
    ensembleRegressor.fit(X_train, y_train,cat_features = cat_features, use_best_model=True)

    print("Finished training")

    # y_pred = test["minutes_until_departure"] - 15  
    # print("Baseline:", mean_absolute_error(y_test, y_pred))

    # y_pred = regressor.predict(X_train)
    # print("Regression tree train error:", mean_absolute_error(y_train, y_pred))

    # Remove the evaluation of the model
    # y_pred = ensembleRegressor.predict(X_test)
    # print("Ensemble of tree regressors test error:", mean_absolute_error(y_test, y_pred))
    filename = f'model_w_mfs_lamp_time_etd_{airport}.sav'
    pickle.dump(ensembleRegressor, open(OUTPUT_DIRECTORY / filename, 'wb'))
    print("Saved the model for the airport: ", airport)