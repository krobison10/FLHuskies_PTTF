#
# Authors: Yudong Lin, Kyler Robison
# Basic run of various models with the new validation technique.
#

import argparse
from datetime import datetime

import lightgbm  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import xgboost  # type: ignore
import mytools
import pandas as pd
from constants import ALL_AIRPORTS, TARGET_LABEL
from sklearn.metrics import mean_absolute_error  # type: ignore

# mytools.ModelRecords.display_best()

if __name__ == "__main__":
    hyperparameter: dict = {
        "num_leaves": 1024 * 4,
        "num_iterations": 128,
        "boosting_type": "gbdt",
        "device_type": "gpu",
        "gpu_use_dp": True,
    }

    # using argparse to parse the argument from command line
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("-a", help="airport")
    parser.add_argument("-o", help="override")
    parser.add_argument("-g", help="enable gpu")
    parser.add_argument("-m", help="use xgboost data")
    args: argparse.Namespace = parser.parse_args()

    airports: tuple[str, ...] = ALL_AIRPORTS if args.a is None else (str(args.a).upper(),)

    train_mae: float = 0.0
    val_mae: float = 0.0

    for airport in airports:
        # load train and test data frame
        train_df, val_df = mytools.get_train_and_test_ds(airport)

        X_train: pd.DataFrame = train_df.drop(columns=[TARGET_LABEL])
        X_test: pd.DataFrame = val_df.drop(columns=[TARGET_LABEL])

        y_train: pd.Series = train_df[TARGET_LABEL]
        y_test: pd.Series = val_df[TARGET_LABEL]

        # check if the same hyperparameter has been used before
        same_setup_mae: float = -1
        if args.o is None or str(args.o).lower().startswith("f"):
            for value in mytools.ModelRecords.get(airport).values():
                if (
                    value.get("num_leaves") == hyperparameter.get("num_leaves")
                    and value.get("num_iterations") == hyperparameter.get("num_iterations")
                    and value.get("boosting_type") == hyperparameter.get("boosting_type")
                    and sorted(value.get("features", [])) == sorted(X_train.columns.to_list())
                ):
                    same_setup_mae = value["val_mae"]
                    break
            if same_setup_mae > 0:
                print(f"Same setup found for airport {airport} found with mae {same_setup_mae}, skip!")
                continue

        if str(args.m).lower().startswith("t"):
            train_matrix: xgboost.DMatrix = xgboost.DMatrix(X_train, enable_categorical=True)
            test_matrix: xgboost.DMatrix = xgboost.DMatrix(X_test, enable_categorical=True)
            xgboost_model = xgboost.XGBRegressor()
            xgboost_model.load_model(mytools.get_model_path(f"xgboost_regression_{airport}.json"))
            X_train["xgboost_pred"] = pd.Series(xgboost_model.predict(X_train))
            X_test["xgboost_pred"] = pd.Series(xgboost_model.predict(X_test))

        # train model
        params: dict[str, str | int | float] = {
            "objective": "regression_l1",
            "learning_rate": 0.05,
            "verbosity": -1,
        }
        params.update(hyperparameter)

        model = lightgbm.train(params, lightgbm.Dataset(X_train, label=y_train))

        y_pred = model.predict(X_train)
        train_mae = round(mean_absolute_error(y_train, y_pred), 4)
        print(f"MAE on train data {airport}: {train_mae}")
        y_pred = model.predict(X_test)
        val_mae = round(mean_absolute_error(y_test, y_pred), 4)
        print(f"MAE on validation data {airport}: {val_mae}")

        # note down model information
        model_record_latest: dict = {
            "train_mae": train_mae,
            "val_mae": val_mae,
            "features": X_train.columns.tolist(),
        }

        # if the model is the best, the save it
        model_record_current_best: dict | None = mytools.ModelRecords.get_smallest(airport)
        if model_record_current_best is None or model_record_current_best["val_mae"] > model_record_latest["val_mae"]:
            if model_record_current_best is not None:
                print(f'The best result so far (previous best: {model_record_current_best["val_mae"]}), saved!')
            else:
                print(f"The best result so far, saved!")
            mytools.save_model(airport, model)
        else:
            print(f'Worse than previous best: {model_record_current_best["val_mae"]})')

        # update record
        model_name: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_record_latest.update(hyperparameter)
        mytools.ModelRecords.update(airport, model_name, model_record_latest)

        # save the latest records
        mytools.ModelRecords.save()

        # plot and save the graph that shows importance
        lightgbm.plot_importance(model, ignore_zero=False, importance_type="gain")
        plt.savefig(mytools.get_model_path(f"lgbm_{airport}_{model_name}_gain_importance.png"), bbox_inches="tight")
        lightgbm.plot_importance(model, ignore_zero=False, importance_type="split")
        plt.savefig(mytools.get_model_path(f"lgbm_{airport}_{model_name}_split_importance.png"), bbox_inches="tight")

    if airport == "ALL":
        global_model = mytools.get_model(airport)
        for theAirport in ALL_AIRPORTS[:10]:
            train_df, val_df = mytools.get_train_and_test_ds(theAirport)

            X_train, X_test = train_df.drop(columns=[TARGET_LABEL]), val_df.drop(columns=[TARGET_LABEL])
            y_train, y_test = train_df[TARGET_LABEL], val_df[TARGET_LABEL]

            train_mae = round(mean_absolute_error(global_model.predict(X_train), y_train), 4)
            val_mae = round(mean_absolute_error(global_model.predict(X_test), y_test), 4)

            print(f"--------------------------------------------------")
            print(f"Apply cumulative model on {theAirport}: train - {train_mae}, test - {val_mae}")
            individual_model_best_mae = mytools.ModelRecords.get_smallest(theAirport)["val_mae"]
            print(f"Compare to individual model's best current best {individual_model_best_mae},")
            if individual_model_best_mae > val_mae:
                print("Cumulative model is better.")
            elif individual_model_best_mae == val_mae:
                print("They are the same.")
            else:
                print("Individual model is better.")
