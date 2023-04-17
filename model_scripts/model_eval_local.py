# Daniil Filienko
#
# Running the model emsemble with Kyler's Train Split for Trevor
# to attain accuracy values for individual airports and overall

from Yudong_scripts.mytools import *
import matplotlib.pyplot as plt
from train_test_split import *
import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns
from lightgbm import LGBMRegressor, Dataset
import lightgbm as lgb
import pickle
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_absolute_error
import argparse
from pathlib import Path

# ---------------------------------------- MAIN ----------------------------------------
DATA_DIRECTORY = Path("full_tables")
OUTPUT_DIRECTORY = Path("./models/Daniil_models")

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


parser: argparse.ArgumentParser = argparse.ArgumentParser()
parser.add_argument("-s", help="save the model")
args: argparse.Namespace = parser.parse_args()

def plotImp(model, X, airport = "ALL", num = 20, fig_size = (40, 20)):
    feature_imp = pd.DataFrame({'Value':model.feature_importances_,'Feature':X.columns})
    plt.figure(figsize=fig_size)
    sns.set(font_scale = 1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", 
                                                        ascending=False)[0:num])
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig(f'lgbm_importances_{airport}_local.png')

y_tests = [0]
y_preds = [0]
for airport in airports:
    # replace this path with the locations of the full tables for each airport if necessary
    df = pd.read_csv(DATA_DIRECTORY / f"{airport}_full.csv",parse_dates=["gufi_flight_date","timestamp"])
    # df.rename(columns = {'wind_direction':'wind_direction_cat', 'cloud_ceiling':'cloud_ceiling_cat', 'visibility':'visibility_cat'}, inplace = True)
    val_df = pd.DataFrame()
    
    train_df = df
    df['precip'] = df['precip'].astype(str)

    #Doing Ordinal Encoding for specified features
    ENCODER: dict[str, OrdinalEncoder] = get_encoder(airport, train_df, val_df)
    for col in ENCODED_STR_COLUMNS:
        if(col != "year"):
            train_df[[col]] = ENCODER[col].transform(train_df[[col]])

    cat_features = get_clean_categorical_columns()
    for c in train_df.columns:
        if any(c in x for x in cat_features):
            train_df[c] = train_df[c].astype('category')

    features = ['minutes_until_etd', 'deps_3hr', 'arrs_3hr', 'deps_taxiing', 'arrs_taxiing', 'exp_deps_15min', 'exp_deps_30min', 'delay_3hr', 'standtime_3hr', 'dep_taxi_3hr', 'arr_taxi_3hr', '1h_ETDP', 'departure_runways', 'arrival_runways', 'temperature', 'wind_direction', 'wind_speed', 'wind_gust', 'cloud_ceiling', 'cloud', 'lightning_prob', 'gufi_flight_major_carrier', 'gufi_flight_destination_airport', 'gufi_timestamp_until_etd', 'year', 'month', 'day', 'hour', 'minute'
            ]

    # evaluating individual airport accuracy
    print(f"Training LIGHTGBM model for {airport}\n")
    X_train = (train_df[features])
    y_train = (train_df["minutes_until_pushback"])
    train_data = lgb.Dataset(X_train, label=y_train)

    fit_params={ 
        'objective': 'regression_l1', # Type of task (regression)
        'metric': 'mae', # Evaluation metric (mean squared error)
        'num_leaves': 1024 * 8,
        'n_estimators': 128,
        }
    
    regressor = LGBMRegressor(**fit_params)

    regressor.fit(X_train, y_train)

    # # SAVING THE MODEL
    save_table_as: str = "save" if args.s is None else str(args.s)
    if save_table_as == "save":
        filename = f'model_{airport}_yudong.sav'
        pickle.dump(regressor, open(OUTPUT_DIRECTORY / filename, 'wb'))
        print("Saved the model for the airport: ", airport)


print(features)
# y_tests = np.hstack(y_tests)
# y_pred = np.hstack(y_preds)
print(f"MAE on all test data: {mean_absolute_error(y_tests, y_preds):.4f}\n")


exit()