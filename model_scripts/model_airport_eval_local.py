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

def plot_feature_importance(model, feature_names, max_num_features=10, importance_type='split', airport = "ALL", fig_size = (40, 20), airline = "ALL"):
    """
    Plots a graph of feature importance for a LightGBM model.

    Parameters:
        -- model: LightGBM model object
            The trained LightGBM model.
        -- feature_names: list or array-like
            List of feature names used in the model.
        -- max_num_features: int, optional (default=10)
            Maximum number of top features to display in the graph.
        -- importance_type: str, optional (default='split')
            Type of feature importance to use. Must be one of {'split', 'gain'}.
        -- figsize: tuple, optional (default=(8, 6))
            Figure size of the plot.

    Returns:
        None
    """

    # Get feature importance values
    importance_vals = model.feature_importance(importance_type=importance_type)
    
    # Get feature names
    feature_names = np.array(feature_names)

    # Sort feature importance values and corresponding feature names in descending order
    sorted_idx = np.argsort(importance_vals)[::-1]
    sorted_importance_vals = importance_vals[sorted_idx]
    sorted_feature_names = feature_names[sorted_idx]

    # Truncate to maximum number of features
    sorted_importance_vals = sorted_importance_vals[:max_num_features]
    sorted_feature_names = sorted_feature_names[:max_num_features]

    # Create a bar plot of feature importance
    plt.figure(figsize=fig_size)
    plt.barh(range(len(sorted_importance_vals)), sorted_importance_vals)
    plt.yticks(range(len(sorted_importance_vals)), sorted_feature_names)
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance for LightGBM Model')
    plt.savefig(f'lgbm_importances_{airport}_{airline}.png')
    
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
            # 'boosting_type': 'gbdt', # Type of boosting algorithm
            'objective': 'regression_l1', # Type of task (regression)
            'metric': 'mae', # Evaluation metric (mean squared error)
            'learning_rate': 0.02, # Learning rate for boosting
            'verbose': 0, # Verbosity level (0 for silent)
            'n_estimators': 4000
        }

        regressor = lgb.train(params, train_data)

        y_pred = regressor.predict(X_test)

        # fit_params={ 
        #             "eval_metric" : 'MAE', 
        #             'verbose': 100,
        #             'feature_name': 'auto', # that's actually the default
        #             'categorical_feature': 'auto' # that's actually the default
        #         }

        # ensembleRegressor = LGBMRegressor(objective="regression_l1", boosting_type='rf')

        # ensembleRegressor.fit(X_train, y_train, **fit_params)

        # ensembleRegressor.fit(X_train, y_train,cat_features=cat_features,use_best_model=True)

        # ensembleRegressor.fit(X_train, y_train, **fit_params)
        # y_pred = ensembleRegressor.predict(X_test,num_iteration=ensembleRegressor.best_iteration_)

        print("Finished training")
        print(f"MAE for {airline} at {airport} test data: {mean_absolute_error(y_test, y_pred):.4f}\n")
        # appending the predictions and test to a single datasets to evaluate overall performance
        
        y_tests = np.concatenate((y_tests, y_test))
        y_preds = np.concatenate((y_preds, y_pred))
        # X_tests = np.concatenate((X_tests, X_test))
        plot_feature_importance(regressor,features,airport=airport, airline=airline)

        # # # SAVING THE MODEL
        # save_table_as: str = "no_save" if args.s is None else str(args.s)
        # if save_table_as != "no_save":
        #     filename = f'model_{airport}.sav'
        #     pickle.dump(regressor, open(OUTPUT_DIRECTORY / filename, 'wb'))
        #     print(f"Saved the model for the {airport} at {airline}")
    
    plot_feature_importance(regressor,features,airport=airport, airline=airline)
    print(f"MAE on {airport} test data: {mean_absolute_error(y_tests, y_preds):.4f}\n")

    # # # SAVING THE MODEL
    # save_table_as: str = "no_save" if args.s is None else str(args.s)
    # if save_table_as != "no_save":
    #     filename = f'model_{airport}.sav'
    #     pickle.dump(regressor, open(OUTPUT_DIRECTORY / filename, 'wb'))
    #     print("Saved the model for the airport: ", airport)


plot_feature_importance(regressor,features,airport=airport, airline=airline)
print(f"MAE on all test data: {mean_absolute_error(y_tests, y_preds):.4f}\n")


exit()