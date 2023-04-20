#
# Author: Kyler Robison
# Modified by: Yudong Lin
# Basic run of various models with the new validation technique.
#

import argparse
import json
import os
from datetime import datetime

import lightgbm  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import mytools
import pandas as pd  # type: ignore
from sklearn.metrics import mean_absolute_error  # type: ignore
from sklearn.preprocessing import OrdinalEncoder  # type: ignore

if __name__ == "__main__":
    hyperparameter: dict = {
        "num_leaves": 1024 * 4,
        "n_estimators": 128,
        "boosting_type": "gbdt",
    }

    # using argparse to parse the argument from command line
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("-a", help="airport")
    parser.add_argument("-o", help="override")
    args: argparse.Namespace = parser.parse_args()

    airports: tuple[str, ...] = mytools.ALL_AIRPORTS if args.a is None else (str(args.a).upper(),)

    model_records_path: str = mytools.get_model_path(f"model_records.json")
    model_records: dict[str, dict] = {}
    if os.path.exists(model_records_path):
        with open(model_records_path, "r", encoding="utf-8") as f:
            model_records = dict(json.load(f))

    TARGET_LABEL: str = "minutes_until_pushback"

    for airport in airports:
        # check if the same hyperparameter has been used before
        same_setup_mae: float = -1
        if airport not in model_records:
            model_records[airport] = {}
        elif args.o is None or str(args.o).lower().startswith("f"):
            for value in model_records[airport].values():
                if (
                    value["num_leaves"] == hyperparameter["num_leaves"]
                    and value["n_estimators"] == hyperparameter["n_estimators"]
                    and value["boosting_type"] == hyperparameter["boosting_type"]
                    and sorted(value["ignore_features"]) == mytools.get_ignored_features()
                ):
                    same_setup_mae = value["mae"]
                    break
            if same_setup_mae > 0:
                print(f"Same setup found for airport {airport} found with mae {same_setup_mae}, skip!")
                continue

        # load data
        train_df: pd.DataFrame = mytools.get_train_tables(airport, remove_duplicate_gufi=False)
        val_df: pd.DataFrame = mytools.get_validation_tables(airport, remove_duplicate_gufi=False)

        # need to make provisions for handling unknown values
        ENCODER: dict[str, OrdinalEncoder] = mytools.get_encoder(airport, train_df, val_df)
        for col in mytools.ENCODED_STR_COLUMNS:
            train_df[[col]] = ENCODER[col].transform(train_df[[col]])
            val_df[[col]] = ENCODER[col].transform(val_df[[col]])
        for col in mytools.get_categorical_columns():
            train_df[col] = train_df[col].astype("category")
            val_df[col] = val_df[col].astype("category")

        # drop useless columns
        train_df.drop(columns=mytools.get_ignored_features(), inplace=True)
        val_df.drop(columns=mytools.get_ignored_features(), inplace=True)

        X_train = train_df.drop(columns=[TARGET_LABEL])
        X_test = val_df.drop(columns=[TARGET_LABEL])

        y_train = train_df[TARGET_LABEL]
        y_test = val_df[TARGET_LABEL]

        # train model
        params = {"objective": "regression_l1", "device_type": "gpu", "learning_rate": 0.05}
        params.update(hyperparameter)

        model = lightgbm.train(params, lightgbm.Dataset(X_train, label=y_train))

        y_pred = model.predict(X_train)
        mae: float = round(mean_absolute_error(y_train, y_pred), 4)
        print(f"MAE on train data {airport}: {mae}")
        y_pred = model.predict(X_test)
        mae = round(mean_absolute_error(y_test, y_pred), 4)
        print(f"MAE on validation data {airport}: {mae}")

        # record model information
        model_name: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_records[airport][model_name] = {
            "mae": mae,
            "ignore_features": mytools.get_ignored_features(),
            "features": X_train.columns.tolist(),
        }
        model_records[airport][model_name].update(hyperparameter)

        # plot the graph that shows importance
        lightgbm.plot_importance(model, ignore_zero=False, importance_type="gain")
        plt.savefig(mytools.get_model_path(f"lgbm_{airport}_{model_name}_gain_importance.png"), bbox_inches="tight")
        lightgbm.plot_importance(model, ignore_zero=False, importance_type="split")
        plt.savefig(mytools.get_model_path(f"lgbm_{airport}_{model_name}_split_importance.png"), bbox_inches="tight")

        # if the model is the best, the save it
        if (
            "best" not in model_records[airport]
            or model_records[airport]["best"]["mae"] > model_records[airport][model_name]["mae"]
        ):
            if "best" in model_records[airport]:
                print(f'The best result so far (previous best: {model_records[airport]["best"]["mae"]}), saved!')
            model_records[airport]["best"] = model_records[airport][model_name]
            model_records[airport]["best"]["achieve_at"] = model_name
            mytools.save_model(airport, model)
        else:
            print(f'Worse than previous best: {model_records[airport]["best"]["mae"]})')

        with open(model_records_path, "w", encoding="utf-8") as f:
            json.dump(model_records, f, indent=4, ensure_ascii=False, sort_keys=True)

        if airport == "ALL":
            for theAirport in mytools.ALL_AIRPORTS[:10]:
                val_airport_df: pd.DataFrame = val_df.loc[val_df.airport == theAirport]
                X_test = val_airport_df.drop(columns=[TARGET_LABEL])
                y_test = val_airport_df[TARGET_LABEL]
                y_pred = model.predict(X_test)
                mae = round(mean_absolute_error(y_test, y_pred), 4)
                print(f"--------------------------------------------------")
                print(f"MAE when apply cumulative model on validation data for airport {theAirport}: {mae}")
                individual_model_best_mae = model_records[theAirport]["best"]["mae"]
                print(f"Compare to individual model's best current best {individual_model_best_mae},")
                if individual_model_best_mae > mae:
                    print("Cumulative model is better.")
                elif individual_model_best_mae == mae:
                    print("They are the same.")
                else:
                    print("Individual model is better.")
