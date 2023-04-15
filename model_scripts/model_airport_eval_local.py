# Daniil Filienko
#
# Running the model emsemble with Kyler's Train Split for Trevor
# to attain accuracy values for individual airports and overall

import matplotlib.pyplot as plt
from train_test_split import *
import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns
from lightgbm import LGBMRegressor
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

def filter_dataframes_by_column(table, column_name):
    """
    Filters a table of data based on the 'carrier' column and returns an array of filtered dataframes,
    where each dataframe contains data for a unique carrier.
    
    Args:
        table (pd.DataFrame): Table of data containing the 'gufi_flight_major_carrier', 'etd', 'carrier', 
                              and 'gufi_flight_destination_airport' columns.
                              
    Returns:
        list: An array of filtered dataframes, where each dataframe contains data for a unique carrier.
    """
    # Create a dictionary to store the dataframes
    carrier_dataframes = {}

    # Get unique values in the 'carrier' column
    unique_carriers = table[column_name].unique()

    # Iterate through unique carriers and create dataframes for each carrier
    for carrier in unique_carriers:
        # Filter the table based on carrier
        filtered_df = table[table[column_name] == carrier]

        # Store the filtered dataframe in the dictionary with carrier as the key
        carrier_dataframes[carrier] = filtered_df
        
    return carrier_dataframes

parser: argparse.ArgumentParser = argparse.ArgumentParser()
parser.add_argument("-s", help="save the model")
args: argparse.Namespace = parser.parse_args()

carrier: str = "major" if args.s is None else str(args.s)


def plotImp(model, X, airport = "ALL", airline = "ALL",num=20, fig_size=(40, 20)):
    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": X.columns})
    plt.figure(figsize=fig_size)
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title("LightGBM Features (avg over folds)")
    plt.tight_layout()
    plt.savefig(f"lgbm_importances_{airline}_at_{airport}_local.png")

y_tests = [0]
y_preds = [0]
# X_tests = [0]
for airport in airports:
    # replace this path with the locations of the full tables for each airport if necessary
    df = pd.read_csv(DATA_DIRECTORY / f"{airport}_full.csv",parse_dates=["gufi_flight_date","timestamp"])
    # df.rename(columns = {'wind_direction':'wind_direction_cat', 'cloud_ceiling':'cloud_ceiling_cat', 'visibility':'visibility_cat'}, inplace = True)

    train, val = split(table=df, airport=airport, save=False)
    
    for c in train.columns:
        col_type = train[c].dtype
        if col_type == 'object' or col_type == 'string' or "cat" in c:
            train[c] = train[c].astype('category')
    for c in val.columns:
        col_type = val[c].dtype
        if col_type == 'object' or col_type == 'string' or "cat" in c:
            val[c] = val[c].astype('category')

    if carrier != "major":
        train_dfs = filter_dataframes_by_column(train,"major_carrier")
        val_dfs = filter_dataframes_by_column(val,"major_carrier")
        carrier_column_name = "major_carrier"
    else:
        train_dfs = filter_dataframes_by_column(train,"gufi_flight_major_carrier")
        val_dfs = filter_dataframes_by_column(val,"gufi_flight_major_carrier")
        carrier_column_name = "gufi_flight_major_carrier"



    offset = 2
    features_all = (df.columns.values.tolist())[offset:(len(df.columns.values))]
    features_remove = ("gufi_flight_date","minutes_until_pushback")
    features = [x for x in features_all if x not in features_remove]
        
    airlines_train = train[carrier_column_name].unique()
    airlines_val = val[carrier_column_name].unique()

    for airline in airlines_train:    
        train = train_dfs[airline]
        X_train = train[features]
        y_train = train["minutes_until_pushback"]
        
        train_data = lgb.Dataset(X_train, label=y_train)

        # Hyperparameters
        # params = {
        # # 'boosting_type': 'gbdt', # Type of boosting algorithm
        # 'objective': 'regression_l1', # Type of task (regression)
        # 'metric': 'mae', # Evaluation metric (mean squared error)
        # 'learning_rate': 0.02, # Learning rate for boosting
        # 'verbose': 0, # Verbosity level (0 for silent)
        # 'n_estimators': 4000
        # }

        # regressor = lgb.train(params, train_data)

        fit_params = { 
            'objective': 'regression_l1', # Type of task (regression)
            'metric': 'mae', # Evaluation metric (mean squared error)
            "n_estimators": 4000,
            "learning_rate":0.02
        }

        regressor = LGBMRegressor(**fit_params)

        regressor.fit(X_train, y_train)

        filename = f'model_{airline}_at_{airport}_gufi.sav'
        pickle.dump(regressor, open(OUTPUT_DIRECTORY / filename, 'wb'))
        print(f"Saved the model for {airline} at: ", airport)
       

    for airline in airlines_val:
        if airline not in train[carrier_column_name].values:
            #Replace the unknown value with the most frequently [assuming best trained] model
            pickled_airline = train[carrier_column_name].mode().iloc[0]

        val = val_dfs[airline]

        X_val = val[features]
        y_val = val["minutes_until_pushback"]

        # open file where we stored the pickled model
        filename = f'model_{airline}_at_{airport}_gufi.sav'
        regressor = pickle.load(open(OUTPUT_DIRECTORY / filename, 'rb'))

        y_pred = regressor.predict(X_val)

        y_tests = np.concatenate((y_tests, y_val))
        y_preds = np.concatenate((y_preds, y_pred))
        print(f"Regression tree train error for {airline} at {airport}:", mean_absolute_error(y_pred,y_val))
        # plotImp(regressor,X_val, airline=airline)

print(f"MAE on all test data: {mean_absolute_error(y_tests, y_preds):.4f}\n")


exit()