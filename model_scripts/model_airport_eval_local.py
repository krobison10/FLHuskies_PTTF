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

    train_df, val_df = split(table=df, airport=airport, save=False)
    
    for c in train_df.columns:
        col_type = train_df[c].dtype
        if col_type == 'object' or col_type == 'string' or "cat" in c:
            train_df[c] = train_df[c].astype('category')
    for c in val_df.columns:
        col_type = val_df[c].dtype
        if col_type == 'object' or col_type == 'string' or "cat" in c:
            val_df[c] = val_df[c].astype('category')

    if carrier == "major":
        train_dfs = filter_dataframes_by_column(train_df,"major_carrier")
        val_dfs = filter_dataframes_by_column(val_df,"major_carrier")
    else:
        train_dfs = filter_dataframes_by_column(train_df,"gufi_flight_major_carrier")
        val_dfs = filter_dataframes_by_column(val_df,"gufi_flight_major_carrier")


    offset = 2
    features_all = (train_df.columns.values.tolist())[offset:(len(train_df.columns.values))]
    features_remove = ("gufi_flight_date","minutes_until_pushback")
    features = [x for x in features_all if x not in features_remove]
    
    for airline in train_dfs.keys():
        train_df = train_dfs[airline]
        val_df = val_dfs[airline]
        # evaluating individual airport accuracy
        print(f"Training LIGHTGBM model for {airline} at {airport}\n")
        X_train = (train_df[features])
        X_test = (val_df[features])
        y_train = (train_df["minutes_until_pushback"])
        y_test = (val_df["minutes_until_pushback"])
        train_data = lgb.Dataset(X_train, label=y_train)

        params = {
            'objective': 'regression_l1', # Type of task (regression)
            'metric': 'mae', # Evaluation metric (mean squared error)
            "n_estimators":4000,
            "learning_rate":0.02
        }

        regressor = lgb.train(params, train_data)

        y_pred = regressor.predict(X_test)

        print("Finished training")
        print(f"MAE for {airline} at {airport} test data: {mean_absolute_error(y_test, y_pred):.4f}\n")
       
        # appending the predictions and test to a single datasets to evaluate overall performance
        y_tests = np.concatenate((y_tests, y_test))
        y_preds = np.concatenate((y_preds, y_pred))
        # plotImp(regressor,X_test,airport=airport, airline=airline)

        filename = f'model_{airline}_{airport}.sav'
        pickle.dump(regressor, open(OUTPUT_DIRECTORY / filename, 'wb'))
        print(f"Saved the model for the {airport} at {airline}")
    
    print(f"MAE on {airport} test data: {mean_absolute_error(y_tests, y_preds):.4f}\n")

print(f"MAE on all test data: {mean_absolute_error(y_tests, y_preds):.4f}\n")


exit()