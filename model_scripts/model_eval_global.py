# @author:Daniil Filienko
from pathlib import Path
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import lightgbm as lgb
import pickle
import argparse

DATA_DIRECTORY_TRAIN = Path("./train_tables")
DATA_DIRECTORY_VAL = Path("./validation_tables")
OUTPUT_DIRECTORY = Path("./models/Daniil_models")

parser: argparse.ArgumentParser = argparse.ArgumentParser()
parser.add_argument("-s", help="save the model")
args: argparse.Namespace = parser.parse_args()

def plotImp(model, X, airport = "ALL", num=20, fig_size=(40, 20)):
    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": X.columns})
    plt.figure(figsize=fig_size)
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title("LightGBM Features (avg over folds)")
    plt.tight_layout()
    plt.savefig(f"lgbm_importances_{airport}_global.png")

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
val = pd.read_csv(DATA_DIRECTORY_VAL / f"ALL_validation.csv", parse_dates=["gufi_flight_date","timestamp"])

#Split into train and test datasets
# test = train.iloc[round(train.shape[0]*0.99):]
# train = train.iloc[:round(train.shape[0]*0.1)]

# make sure that the categorical features are encoded as strings
# cat_feature = train.columns[np.where(train.dtypes != float)[0]].values.tolist()
# train[cat_feature] = train[cat_feature].astype(str)

# train.rename(columns = {'wind_direction':'wind_direction_cat', 'cloud_ceiling':'cloud_ceiling_cat', 'visibility':'visibility_cat'}, inplace = True)

for c in train.columns:
    col_type = train[c].dtype
    if col_type != 'float' or col_type == 'string' in c:
        train[c] = train[c].astype('category')

for c in val.columns:
    col_type = val[c].dtype
    if col_type == 'object' or col_type == 'string' in c:
        val[c] = val[c].astype('category')

#remove test for training the models
# test[cat_feature] = test[cat_feature].astype(str)

print("Generated a shared dataframe")

# Preventing GUFI from being an attribute to analyze
offset = 2
features_all = (train.columns.values.tolist())[offset:(len(train.columns.values))]
features_all_val = (val.columns.values.tolist())[offset:(len(val.columns.values))]

print("Difference, in train but not val: ",[x for x in features_all if x not in features_all_val])
print("Difference, in val but not train: ",[x for x in features_all_val if x not in features_all])

features_remove = ("gufi_flight_date","minutes_until_pushback")
features = [x for x in features_all if x not in features_remove]
features_val = ["minutes_until_pushback","airport"]

X_train = train[features]
y_train = train[features_val]

X_val = val[features]
y_val = val[features_val]

print("Train:",X_train.shape)
print("Val:",X_val.shape)

label = y_train["minutes_until_pushback"]

train_data = lgb.Dataset(X_train, label=label)

fit_params = { 
    'boosting_type': 'rf', # Type of boosting algorithm
    'objective': 'regression_l1', # Type of task (regression)
    'metric': 'mae', # Evaluation metric (mean squared error)
    'learning_rate': 0.02, # Learning rate for boosting
    'verbose': 0, # Verbosity level (0 for silent)
    'n_estimators': 4000
}

regressor = LGBMRegressor(**fit_params)

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_val,num_iteration=regressor.best_iteration_)

# params = {
#     # 'boosting_type': 'rf', # Type of boosting algorithm
#     'objective': 'regression_l1', # Type of task (regression)
#     'metric': 'mae', # Evaluation metric (mean squared error)
#     'learning_rate': 0.02, # Learning rate for boosting
#     'verbose': 0, # Verbosity level (0 for silent)
#     'n_estimators': 4000
# }
# regressor = lgb.train(params, train_data)

# y_pred = regressor.predict(X_val)

print(f"Regression tree train error for ALL:", mean_absolute_error(y_val["minutes_until_pushback"], y_pred))
plotImp(regressor, X_val)

for airport in AIRPORTS:
    print(len(AIRPORTS))
    X_val_local = X_val.loc[X_val['airport'] == airport]
    y_val_local = y_val.loc[y_val['airport'] == airport]

    y_pred = regressor.predict(X_val_local)
    plotImp(regressor, X_val_local, airport=airport)
    print(f"Regression tree train error for {airport}:", mean_absolute_error(y_val_local["minutes_until_pushback"], y_pred))

# Remove the evaluation of the model
# y_pred = ensembleRegressor.predict(X_test)
# print("Ensemble of tree regressors test error:", mean_absolute_error(y_test, y_pred))


# # SAVING THE MODEL
save_table_as: str = "save" if args.s is None else str(args.s)
if save_table_as == "save":
    filename = f'model_GLOBAL.sav'
    pickle.dump(regressor, open(OUTPUT_DIRECTORY / filename, 'wb'))
    print("Saved the model for the airport: ", airport)