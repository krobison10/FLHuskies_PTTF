from sys import argv
import time
import pandas as pd
import catboost as cb

from src.GeneralUtilities import *
from src.CreateMaster import CreateMaster

start_time = pd.to_datetime("2021-03-23")
end_time = pd.to_datetime("2021-03-25")


def main(data_path, models_path, build_master=False):
    """
    Main script that loads the raw data, creates the master table at the airport-timestamp level,
    adds test indicators and builds 120 regressor models, one for each combination of airport
    and lookahead period and stores them in the models path

    :params str data_path: Path where the raw data is located
    :params str models_path: Path where the models will be stored
    :params Bool build_master: Indicator of whether to compute the master or read it

    :returns: None - models are saved in models_path
    """

    if build_master:

        # Create the master table at the airport and timestamp level
        print(time.ctime(), "Creating master table")
        master_table = CreateMaster(
            data_path=data_path,
            airports=AIRPORTS,
            start_time=start_time,
            end_time=end_time,
            with_targets=True,
        )

        # Store the master table
        print(time.ctime(), "Saving master table")
        master_table.to_parquet(f"{data_path}/master_table_v3.parquet")

    else:
        print(time.ctime(), "Loading master table")
        master_table = pd.read_parquet(f"{data_path}/master_table_v3.parquet")

def performEvaluation():
    
    # Add test indicator in the master table based on the submission format file
    print(time.ctime(), "Adding test indicator to the master table")
    sub_format = pd.read_csv(
        f"{data_path}/open_submission_format.csv", parse_dates=["timestamp"]
    )
    sub_format = sub_format.groupby(["airport", "timestamp"]).sum().reset_index()
    sub_format.columns = ["airport", "timestamp", "in_test"]

    master_table["day"] = master_table["timestamp"].apply(lambda x: str(x)[:10])
    master_table = master_table.merge(
        sub_format, how="left", on=["airport", "timestamp"]
    )

    # Define the numerical and categorical features to train with
    print(time.ctime(), "Splitting numerical and categorical features")
    features = ["aircraft_engine_class", "aircraft_type", "major_carrier", "flight_type", "gufi","departure_runway_estimated_time"]
    cat_features = [i for i in range(len(features)) if "_cat_" in features[i]]

    # Build catboost models at the timestamp-airport level for each airport and lookahead
    for airport in AIRPORTS:
        # Filter to stick only with relevant rows
        current = master_table[(master_table.airport == airport)].reset_index(drop=True)
        target = ['minutes_until_pushback']
        # Define size of train vs validation sets
        n = current.shape[0]
        start_test = int(n * 0.90)
        end_test = int(n)

        # Split train and validation sets
        X_train = current[features].iloc[
            lambda x: (x.index < start_test) | (x.index > end_test)
        ]
        y_train = current[target][
            lambda x: (x.index < start_test) | (x.index > end_test)
        ]
        X_val = current[features].iloc[start_test:end_test]
        y_val = current[target][start_test:end_test]

        # Initialize a big score for the loss
        best_score = 200

        # Iterate for different values of eta and depth and stick with the best
        for eta in [0.005, 0.01, 0.02, 0.03, 0.05]:
            for depth in [3, 4, 5, 6, 7]:

                # Initialize CatBoostClassifier
                model = cb.CatBoostRegressor(
                    n_estimators=6000,
                    task_type="GPU",
                    thread_count=-1,
                    loss_function="MAE",
                    cat_features=cat_features,
                )
                # Fit model
                model.fit(
                    X_train,
                    y_train,
                    eval_set=(X_val, y_val),
                    use_best_model=True,
                    verbose=1,
                )

                if model.best_score_["validation"]["MAE"] < best_score:
                    best_score = model.best_score_["validation"]["MAE"]
                    model.save_model(
                        f"{models_path}/version_final/model_{airport}_{target}"
                    )

                print(
                    "@@@@@@@@@@@@@@@",
                    airport,
                    target,
                    eta,
                    depth,
                    model.best_score_["validation"]["MAE"],
                )


if __name__ == "__main__":
    # Read parameters from the command line
    data_path = "_data"
    models_path = "\model_scripts\Daniil_scripts\src"
    build_master = True

    # Execute main script
    main(data_path, models_path, build_master)