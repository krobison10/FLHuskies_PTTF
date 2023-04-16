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
import pickle
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

    ALL_AIRPORTS: tuple[str, ...] = (
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
        # "ALL",
    )

    airports: tuple[str, ...] = ALL_AIRPORTS if args.a is None else (str(args.a).upper(),)

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

        train_df: pd.DataFrame = mytools.get_train_tables(airport, remove_duplicate_gufi=False)
        train_df.drop(columns=mytools.get_ignored_features(), inplace=True)
        val_df: pd.DataFrame = mytools.get_validation_tables(airport, remove_duplicate_gufi=False)
        val_df.drop(columns=mytools.get_ignored_features(), inplace=True)

        # need to make provisions for handling unknown values
        for col in mytools.ALL_ENCODED_STR_COLUMNS:
            encoder: OrdinalEncoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

            encoded_col = encoder.fit_transform(train_df[[col]])
            train_df[[col]] = encoded_col

            encoded_col = encoder.transform(val_df[[col]])
            val_df[[col]] = encoded_col

        X_train = train_df.drop(columns=[TARGET_LABEL])
        X_test = val_df.drop(columns=[TARGET_LABEL])

        y_train = train_df[TARGET_LABEL]
        y_test = val_df[TARGET_LABEL]

        # train model
        params = {"objective": "regression_l1", "verbosity": -1, "learning_rate": 0.05}
        params.update(hyperparameter)

        dtrain = lightgbm.Dataset(X_train, label=y_train, categorical_feature=mytools.get_categorical_columns())

        model = lightgbm.train(params, dtrain)

        y_pred = model.predict(X_train)
        mae: float = round(mean_absolute_error(y_train, y_pred), 4)
        print(f"MAE on train data {airport}: {mae}")
        y_pred = model.predict(X_test)
        mae = round(mean_absolute_error(y_test, y_pred), 4)
        print(f"MAE on validation data {airport}: {mae}")

        # record model information
        model_name: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_records[airport][model_name] = {"mae": mae, "ignore_features": mytools.get_ignored_features()}
        model_records[airport][model_name].update(hyperparameter)

        # plot the graph that shows importance
        lightgbm.plot_importance(model, ignore_zero=False)
        plt.savefig(mytools.get_model_path(f"lgbm_{airport}_{model_name}_importance.png"), bbox_inches="tight")

        # if the model is the best, the save it
        if (
            "best" not in model_records[airport]
            or model_records[airport]["best"]["mae"] > model_records[airport][model_name]["mae"]
        ):
            if "best" in model_records[airport]:
                print(f'The best result so far (previous best: {model_records[airport]["best"]["mae"]}), saved!')
            model_records[airport]["best"] = model_records[airport][model_name]
            model_records[airport]["best"]["achieve_at"] = model_name
            pickle.dump(model, open(mytools.get_model_path(f"lgbm_{airport}_model.pickle"), "wb"))
        else:
            print(f'Worse than previous best: {model_records[airport]["best"]["mae"]})')

        with open(model_records_path, "w", encoding="utf-8") as f:
            json.dump(model_records, f, indent=4, ensure_ascii=False, sort_keys=True)

        if airport == "ALL":
            for theAirport in ALL_AIRPORTS[: len(ALL_AIRPORTS) - 1]:
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
