# Daniil Filienko
#
# Running the model emsemble with Kyler's Train Split
# save and then report accuracy values for 
# individual airports and overall

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
from lightgbm import LGBMRegressor
import seaborn as sns

# ---------------------------------------- MAIN ----------------------------------------
DATA_DIRECTORY_TRAIN = Path("./train_tables")
DATA_DIRECTORY_VAL = Path("./validation_tables")
OUTPUT_DIRECTORY = Path("./models/Daniil_models")
OUTPUT_FIGURES_DIRECTORY = Path("./figures")
DATA_DIRECTORY = Path("full_tables")

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


def plotImp(model, X, airport = "ALL", num=20, fig_size=(40, 20)):
    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": X.columns})
    plt.figure(figsize=fig_size)
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title("LightGBM Features (avg over folds)")
    plt.tight_layout()
    plt.savefig(f"lgbm_importances_{airport}_global.png")

print("Started")
# train = pd.read_csv(DATA_DIRECTORY_TRAIN / f"ALL_train.csv", parse_dates=["gufi_flight_date","timestamp"])
# val = pd.read_csv(DATA_DIRECTORY_VAL / f"ALL_validation.csv", parse_dates=["gufi_flight_date","timestamp"])

df = pd.read_csv(DATA_DIRECTORY / f"ALL_full.csv",parse_dates=["gufi_flight_date","timestamp"])
train, val = split(table=df, save=False)

for c in val.columns:
    col_type = val[c].dtype
    if col_type == 'object' or col_type == 'string' or "cat" in c:
        val[c] = val[c].astype('category')


#remove train for testing the models
if carrier == "major":
    val_dfs = filter_dataframes_by_column(val,"major_carrier")
    carrier_column_name = "major_carrier"

else:
    val_dfs = filter_dataframes_by_column(val,"gufi_flight_major_carrier")
    carrier_column_name = "gufi_flight_major_carrier"

airlines_val = val[carrier_column_name].unique()
print("Generated a shared dataframe")

# Preventing GUFI from being an attribute to analyze
offset = 2
features_all = (df.columns.values.tolist())[offset:(len(df.columns.values))]

features_remove = ("gufi_flight_date","minutes_until_pushback")
features = [x for x in features_all if x not in features_remove]
features_val = ["minutes_until_pushback","airport"]

for airline in airlines_val:
    pickled_airline = airline
    if airline not in train[carrier_column_name].values:
        #Replace the unknown value with the most frequently [assuming best trained] model
        pickled_airline = train[carrier_column_name].mode().iloc[0]

    val = val_dfs[airline]

    X_val = val[features]
    y_val = val[features_val]

    # open file where we stored the pickled model
    filename = f'model_{pickled_airline}.sav'
    regressor = pickle.load(open(OUTPUT_DIRECTORY / filename, 'rb'))

    y_pred = regressor.predict(X_val)

    y_tests = np.concatenate((y_tests, y_val["minutes_until_pushback"]))
    y_preds = np.concatenate((y_preds, y_pred))
    print(f"Regression tree train error for {airline}:", mean_absolute_error(y_pred,y_val["minutes_until_pushback"]))
    plotImp(regressor,X_val, airline=airline)

print(f"Regression tree train error for ALL:", mean_absolute_error(y_tests, y_preds))

# for airport in AIRPORTS:
#     X_val_local = X_val.loc[X_val['airport'] == airport]
#     y_val_local = y_val.loc[y_val['airport'] == airport]

#     y_pred = regressor.predict(X_val_local)
#     plot_feature_importance(regressor,features,airport=airport)
#     print(f"Regression tree train error for {airport}:", mean_absolute_error(y_val_local["minutes_until_pushback"], y_pred))
exit()