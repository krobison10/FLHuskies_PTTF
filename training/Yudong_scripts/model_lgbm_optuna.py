#
# Author: Yudong Lin
#
# A script that will try to find the most ideal hyperparameter for given model and dataset
#
import argparse
import os

import joblib  # type: ignore
import lightgbm as lgb  # type: ignore
import mytools
import optuna
from constants import ALL_AIRPORTS, TARGET_LABEL
from sklearn.metrics import mean_absolute_error  # type: ignore


def _train(trial, _airport, X_train, X_test, y_train, y_test) -> float:
    params: dict[str, str | int | float] = {
        "boosting_type": "gbdt",
        "objective": "regression_l1",
        "device_type": "gpu",
        "verbosity": -1,
        "num_leaves": trial.suggest_int("num_leaves", 100, 1024 * 8),
        "num_iterations": trial.suggest_int("num_iterations", 64, 64 * 3),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0),
        "subsample_for_bin": trial.suggest_int("subsample_for_bin", 200000, 400000),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2),
    }

    model = lgb.train(params, lgb.Dataset(X_train, label=y_train))

    y_pred = model.predict(X_train)
    train_mae: float = round(mean_absolute_error(y_train, y_pred), 4)
    print(f"MAE on train data {_airport}: {train_mae}")
    y_pred = model.predict(X_test)
    test_mae = round(mean_absolute_error(y_test, y_pred), 4)
    print(f"MAE on validation data {_airport}: {test_mae}")

    # record model information
    model_record: dict = {
        "train_mae": train_mae,
        "val_mae": test_mae,
        "features": X_train.columns.values.tolist(),
    }
    model_record.update(params)

    # if the model is the best, the save it
    model_records_ref: dict[str, dict] = mytools.ModelRecords.get(airport)
    if "best" not in model_records_ref or model_records_ref["best"]["val_mae"] > model_record["val_mae"]:
        if "best" in model_records_ref:
            print(f'The best result so far (previous best: {model_records_ref["best"]["val_mae"]}), saved!')
        mytools.ModelRecords.update(airport, "best", model_record)
        mytools.ModelRecords.save()
        mytools.save_model(airport, model)
    else:
        print(f'Worse than previous best: {model_records_ref["best"]["val_mae"]})')

    return test_mae


if __name__ == "__main__":
    # create or load studies
    studies: dict[str, optuna.Study] = {}
    studies_file_path: str = mytools.get_model_path("studies.pkl")
    # if studies file already exists, then load it
    if os.path.exists(studies_file_path):
        studies = joblib.load(studies_file_path)

    for airport in ALL_AIRPORTS:
        # load train and test data frame
        train_df, val_df = mytools.get_train_and_test_ds(airport)

        def _objective(trial) -> float:
            return _train(
                trial,
                airport,
                train_df.drop(columns=[TARGET_LABEL]),
                val_df.drop(columns=[TARGET_LABEL]),
                train_df[TARGET_LABEL],
                val_df[TARGET_LABEL],
            )

        # using argparse to parse the argument from command line
        parser: argparse.ArgumentParser = argparse.ArgumentParser()
        parser.add_argument("-n", help="number of training")
        args: argparse.Namespace = parser.parse_args()

        study: optuna.Study = studies.get(
            airport, optuna.create_study(direction="minimize", study_name=f"{airport}_tuner")
        )
        study.optimize(_objective, n_trials=int(args.n) if args.n is not None else 10)

        # save checkpoint
        studies[airport] = study
        joblib.dump(studies, studies_file_path)

        print("Number of finished trials: {}".format(len(study.trials)))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: {}".format(trial.value))

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
