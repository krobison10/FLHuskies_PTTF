#
# Authors: Yudong Lin, Kyler Robison
# Basic run of various models with the new validation technique.
#

import argparse
from datetime import datetime

import lightgbm  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import mytools
import pandas as pd
from constants import ALL_AIRPORTS, TARGET_LABEL  # type: ignore
from sklearn.metrics import mean_absolute_error  # type: ignore

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

    airports: tuple[str, ...] = ALL_AIRPORTS if args.a is None else (str(args.a).upper(),)

    for airport in airports:
        # load train and test data frame
        train_df, val_df = mytools.get_train_and_test_ds(airport)

        X_train: pd.DataFrame = train_df.drop(columns=[TARGET_LABEL])
        X_test: pd.DataFrame = val_df.drop(columns=[TARGET_LABEL])

        y_train: pd.DataFrame = train_df[TARGET_LABEL]
        y_test: pd.DataFrame = val_df[TARGET_LABEL]

        # check if the same hyperparameter has been used before
        same_setup_mae: float = -1
        if args.o is None or str(args.o).lower().startswith("f"):
            for value in mytools.ModelRecords.get(airport).values():
                if (
                    value.get("num_leaves") == hyperparameter.get("num_leaves")
                    and value.get("n_estimators") == hyperparameter.get("n_estimators")
                    and value.get("boosting_type") == hyperparameter.get("boosting_type")
                    and sorted(value.get("features", [])) == sorted(X_train.columns.to_list())
                ):
                    same_setup_mae = value["val_mae"]
                    break
            if same_setup_mae > 0:
                print(f"Same setup found for airport {airport} found with mae {same_setup_mae}, skip!")
                continue

        """
        train_matrix: xgboost.DMatrix = xgboost.DMatrix(X_train, enable_categorical=True)
        test_matrix: xgboost.DMatrix = xgboost.DMatrix(X_test, enable_categorical=True)

        xgboost_model = xgboost.XGBRegressor()
        xgboost_model.load_model(mytools.get_model_path(f"xgboost_regression_{airport}.json"))

        X_train["xgboost_pred"] = pd.Series(xgboost_model.predict(X_train))
        X_test["xgboost_pred"] = pd.Series(xgboost_model.predict(X_test))
        """

        # train model
        params: dict[str, str | int | float] = {
            "objective": "regression_l1",
            # "learning_rate": 0.05,
            "verbosity": -1,
        }
        params.update(hyperparameter)

        # don not use gpu for global model training due to error
        if airport != "ALL":
            params["device_type"] = "gpu"

        model = lightgbm.train(params, lightgbm.Dataset(X_train, label=y_train))

        y_pred = model.predict(X_train)
        mae: float = round(mean_absolute_error(y_train, y_pred), 4)
        print(f"MAE on train data {airport}: {mae}")
        y_pred = model.predict(X_test)
        mae = round(mean_absolute_error(y_test, y_pred), 4)
        print(f"MAE on validation data {airport}: {mae}")

        # record model information
        model_name: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_info: dict = {
            "val_mae": mae,
            "features": X_train.columns.tolist(),
        }
        model_info.update(hyperparameter)
        mytools.ModelRecords.update(airport, model_name, model_info)

        # plot the graph that shows importance
        lightgbm.plot_importance(model, ignore_zero=False, importance_type="gain")
        plt.savefig(mytools.get_model_path(f"lgbm_{airport}_{model_name}_gain_importance.png"), bbox_inches="tight")
        lightgbm.plot_importance(model, ignore_zero=False, importance_type="split")
        plt.savefig(mytools.get_model_path(f"lgbm_{airport}_{model_name}_split_importance.png"), bbox_inches="tight")

        # if the model is the best, the save it
        model_records_ref: dict[str, dict] = mytools.ModelRecords.get(airport)
        if (
            "best" not in model_records_ref
            or model_records_ref["best"]["val_mae"] > model_records_ref[model_name]["val_mae"]
        ):
            if "best" in model_records_ref:
                print(f'The best result so far (previous best: {model_records_ref["best"]["val_mae"]}), saved!')
            mytools.ModelRecords.update(airport, "best", model_records_ref[model_name])
            mytools.save_model(airport, model)
        else:
            print(f'Worse than previous best: {model_records_ref["best"]["val_mae"]})')

        # save the latest records
        mytools.ModelRecords.save()

        if airport == "ALL":
            for theAirport in ALL_AIRPORTS[:10]:
                all_df: pd.DataFrame = pd.concat(mytools.get_train_and_test_ds(airport), ignore_index=True)
                y_pred = model.predict(all_df.drop(columns=[TARGET_LABEL]))
                mae = round(mean_absolute_error(all_df[TARGET_LABEL], y_pred), 4)
                print(f"--------------------------------------------------")
                print(f"MAE when apply cumulative model on validation data for airport {theAirport}: {mae}")
                individual_model_best_mae = mytools.ModelRecords.get(airport)["best"]["val_mae"]
                print(f"Compare to individual model's best current best {individual_model_best_mae},")
                if individual_model_best_mae > mae:
                    print("Cumulative model is better.")
                elif individual_model_best_mae == mae:
                    print("They are the same.")
                else:
                    print("Individual model is better.")
