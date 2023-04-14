# Daniil Filienko
#
# Running the model emsemble with Kyler's Train Split for Trevor
# to attain accuracy values for individual airports and overall

import matplotlib.pyplot as plt
from train_test_split import *
import pandas as pd
from pathlib import Path
import numpy as np
import lightgbm as lgb
import pickle
from sklearn.metrics import mean_absolute_error
import argparse
from pathlib import Path

# ---------------------------------------- MAIN ----------------------------------------
DATA_DIRECTORY_TRAIN = Path("./train_tables")
DATA_DIRECTORY_VAL = Path("./validation_tables")
OUTPUT_DIRECTORY = Path("./models/Daniil_models")
OUTPUT_FIGURES_DIRECTORY = Path("./figures")

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

print("Started")
train = pd.read_csv(DATA_DIRECTORY_TRAIN / f"ALL_train.csv", parse_dates=["gufi_flight_date","timestamp"])
val = pd.read_csv(DATA_DIRECTORY_VAL / f"ALL_validation.csv", parse_dates=["gufi_flight_date","timestamp"])

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
if carrier == "major":
    train_dfs = filter_dataframes_by_column(train,"major_carrier")
    val_dfs = filter_dataframes_by_column(val,"major_carrier")
else:
    train_dfs = filter_dataframes_by_column(train,"gufi_flight_major_carrier")
    val_dfs = filter_dataframes_by_column(val,"gufi_flight_major_carrier")

print("Generated a shared dataframe")

# Preventing GUFI from being an attribute to analyze
offset = 2
features_all = (train.columns.values.tolist())[offset:(len(train.columns.values))]

features_remove = ("gufi_flight_date","minutes_until_pushback")
features = [x for x in features_all if x not in features_remove]
features_val = ["minutes_until_pushback","airport"]


for airline in train_dfs.keys():
    train = train_dfs[airline]
    val = train_dfs[airline]
    X_train = train[features]
    y_train = train[features_val]

    X_val = val[features]
    y_val = val[features_val]

    train_data = lgb.Dataset(X_train, label=y_train)

    # Hyperparameters
    params = {
    # 'boosting_type': 'gbdt', # Type of boosting algorithm
    'objective': 'regression_l1', # Type of task (regression)
    'metric': 'mae', # Evaluation metric (mean squared error)
    'learning_rate': 0.02, # Learning rate for boosting
    'verbose': 0, # Verbosity level (0 for silent)
    'n_estimators': 4000
    }

    regressor = lgb.train(params, train_data)

    y_pred = regressor.predict(X_val)

    print(f"Regression tree train error for ALL:", mean_absolute_error(y_val["minutes_until_pushback"], y_pred))
    plot_feature_importance(regressor,features, airline=airline)

for airport in AIRPORTS:
    X_val_local = X_val.loc[X_val['airport'] == airport]
    y_val_local = y_val.loc[y_val['airport'] == airport]

    y_pred = regressor.predict(X_val_local)
    plot_feature_importance(regressor,features,airport=airport)
    print(f"Regression tree train error for {airport}:", mean_absolute_error(y_val_local["minutes_until_pushback"], y_pred))

exit()