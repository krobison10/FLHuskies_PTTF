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

    train_df, val_df = split(table=df, airport=airport, save=False)
    
    for c in train_df.columns:
        col_type = train_df[c].dtype
        if col_type == 'object' or col_type == 'string' or "cat" in c:
            train_df[c] = train_df[c].astype('category')
    for c in val_df.columns:
        col_type = val_df[c].dtype
        if col_type == 'object' or col_type == 'string' or "cat" in c:
            val_df[c] = val_df[c].astype('category')

    offset = 2
    features_all = (train_df.columns.values.tolist())[offset:(len(train_df.columns.values))]
    features_remove = ("gufi_flight_date","minutes_until_pushback")
    features = [x for x in features_all if x not in features_remove]

    # evaluating individual airport accuracy
    print(f"Training LIGHTGBM model for {airport}\n")
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
    print(f"MAE on {airport} test data: {mean_absolute_error(y_test, y_pred):.4f}\n")
    # appending the predictions and test to a single datasets to evaluate overall performance
    y_tests = np.concatenate((y_tests, y_test))
    y_preds = np.concatenate((y_preds, y_pred))

    # plotImp(ensembleRegressor,X_test,airport=airport)

print(features)
# y_tests = np.hstack(y_tests)
# y_pred = np.hstack(y_preds)
print(f"MAE on all test data: {mean_absolute_error(y_tests, y_preds):.4f}\n")
exit()