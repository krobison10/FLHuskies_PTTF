# @author:Daniil Filienko
from pathlib import Path
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor, Dataset
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
DATA_DIRECTORY_TRAIN = Path("./train_tables")
DATA_DIRECTORY_VAL = Path("./validation_tables")


def plotImp(model, X, airport, num=20, fig_size=(40, 20)):
    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": X.columns})
    plt.figure(figsize=fig_size)
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title("LightGBM Features (avg over folds)")
    plt.tight_layout()
    plt.savefig(f"lgbm_importances_{airport}_global.png")

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

print("Started")
train = pd.read_csv(DATA_DIRECTORY_TRAIN / f"ALL_train.csv", parse_dates=["gufi_flight_date","timestamp"])
train = train.sort_values(by=['gufi'])

val = pd.read_csv(DATA_DIRECTORY_VAL / f"ALL_validation.csv", parse_dates=["gufi_flight_date","timestamp"])
val = val.sort_values(by=['gufi'])

#Split into train and test datasets
# test = train.iloc[round(train.shape[0]*0.99):]
# train = train.iloc[:round(train.shape[0]*0.1)]

# make sure that the categorical features are encoded as strings
# cat_feature = train.columns[np.where(train.dtypes != float)[0]].values.tolist()
# train[cat_feature] = train[cat_feature].astype(str)

# train.rename(columns = {'wind_direction':'wind_direction_cat', 'cloud_ceiling':'cloud_ceiling_cat', 'visibility':'visibility_cat'}, inplace = True)

for c in train.columns:
    col_type = train[c].dtype
    if col_type == 'object' or col_type == 'string' or "cat" in c:
        train[c] = train[c].astype('category')

for c in val.columns:
    col_type = val[c].dtype
    if col_type == 'object' or col_type == 'string' or "cat" in c:
        val[c] = val[c].astype('category')

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
offset = 2

# cat_features = [10,13,14,15,16,17,18,19,20,21,22,23]
# cat_features = [c - offset for c in cat_features]

features_all = (train.columns.values.tolist())[offset:(len(train.columns.values))]

#For mfs only
# features_remove = ("departure_runway_actual","cloud","aircraft_engine_class","lightning_prob","aircraft_type","major_carrier",
#                                 "flight_type","gufi_end_label","precip")
features_remove = ("gufi_flight_date","minutes_until_pushback")
features = [x for x in features_all if x not in features_remove]
features_val = ["minutes_until_pushback","airport"]
# features_encoded = ["cloud_enc","aircraft_engine_class_enc","lightning_prob_enc","aircraft_type_enc","major_carrier_enc",
#                                "flight_type_enc","gufi_end_label_enc","wind_direction_enc","precip_enc"]
# features = features_all + features_encoded
X_train = train[features]
y_train = train[features_val]

X_val = train[features]
y_val = train[features_val]

# Remove the testing of the features
# X_test = test[features]

# y_test = test["minutes_until_pushback"]
fit_params={ 
            "eval_metric" : 'MAE', 
            'verbose': 100,
            'feature_name': 'auto', # that's actually the default
            'categorical_feature': 'auto' # that's actually the default
        }
ensembleRegressor = LGBMRegressor(objective="regression_l1")
# ensembleRegressor.fit(X_train, y_train,cat_features=cat_features,use_best_model=True)

ensembleRegressor.fit(X_train, y_train["minutes_until_pushback"], **fit_params)

print("Finished training")

# y_pred = test["minutes_until_departure"] - 15  
# print("Baseline:", mean_absolute_error(y_test, y_pred))


y_pred = ensembleRegressor.predict(X_val)
print(f"Regression tree train error for ALL:", mean_absolute_error(y_val["minutes_until_pushback"], y_pred))
plotImp(ensembleRegressor, X_val)

for airport in AIRPORTS:
    X_val_local = X_val.loc[X_val['airport'] == airport]
    y_val_local = y_val.loc[y_val['airport'] == airport]

    y_pred = ensembleRegressor.predict(X_val_local)
    print(f"Regression tree train error for {airport}:", mean_absolute_error(y_val_local["minutes_until_pushback"], y_pred))


# Remove the evaluation of the model
# y_pred = ensembleRegressor.predict(X_test)
# print("Ensemble of tree regressors test error:", mean_absolute_error(y_test, y_pred))



# # SAVING THE MODEL

# filename = f'model_w_mfs_lamp_time_etd_{airport}_lightgmb.sav'
# pickle.dump(ensembleRegressor, open(OUTPUT_DIRECTORY / filename, 'wb'))
# print("Saved the model for the airport: ", airport)