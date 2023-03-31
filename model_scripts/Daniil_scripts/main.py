from sys import argv
import time
import pandas as pd

from src.GeneralUtilities import *
from src.CreateMaster import CreateMaster

start_time = pd.to_datetime("2020-11-02")
end_time = pd.to_datetime("2021-11-01")


def main(data_path, models_path, build_master=False):
    """
    Main script that loads the raw data, creates the master table at the airport-timestamp level,
    adds test indicators and builds 120 multiclass models, one for each combination of airport
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
        master_table.to_parquet(f"{data_path}/master_table_final.parquet")

    else:
        print(time.ctime(), "Loading master table")
        master_table = pd.read_parquet(f"{data_path}/master_table.parquet")

    # # Add test indicator in the master table based on the submission format file
    # print(time.ctime(), "Adding test indicator to the master table")
    # sub_format = pd.read_csv(
    #     f"{data_path}/open_submission_format.csv", parse_dates=["timestamp"]
    # )
    # sub_format = sub_format.groupby(["airport", "timestamp"]).active.sum().reset_index()
    # sub_format.columns = ["airport", "timestamp", "in_test"]

    # master_table["day"] = master_table["timestamp"].apply(lambda x: str(x)[:10])
    # master_table = master_table.merge(
    #     sub_format, how="left", on=["airport", "timestamp"]
    # )

    # # Define the numerical and categorical features to train with
    # print(time.ctime(), "Splitting numerical and categorical features")
    # features = [c for c in master_table.columns if "feat" in c]
    # cat_features = [i for i in range(len(features)) if "_cat_" in features[i]]

    # # Build catboost models at the timestamp-airport level for each airport and lookahead
    # for airport in AIRPORTS:
    #     for target in [f"target_{l}" for l in LOOKAHEADS]:

    #         # Filter to stick only with relevant rows
    #         current = master_table[
    #             (master_table[target].isna() == False)
    #             & (master_table.airport == airport)
    #             & (
    #                 master_table.day.isin(
    #                     master_table[master_table.in_test.isna() == False][
    #                         "day"
    #                     ].unique()
    #                 )
    #                 == False
    #             )
    #         ].reset_index(drop=True)

    #         # Define size of train vs validation sets
    #         n = current.shape[0]
    #         start_test = int(n * 0.95)
    #         end_test = int(n)

    #         # Split train and validation sets
    #         X_train = current[features].iloc[
    #             lambda x: (x.index < start_test) | (x.index > end_test)
    #         ]
    #         y_train = current[target][
    #             lambda x: (x.index < start_test) | (x.index > end_test)
    #         ]
    #         X_val = current[features].iloc[start_test:end_test]
    #         y_val = current[target][start_test:end_test]

    #         # Initialize a big score for the loss
    #         best_score = 200

    #         # Iterate for different values of eta and depth and stick with the best
    #         for eta in [0.005, 0.01, 0.02, 0.03, 0.05]:
    #             for depth in [3, 4, 5, 6, 7]:

    #                 # Initialize CatBoostClassifier
    #                 model = CatBoostClassifier(
    #                     eta=eta,
    #                     n_estimators=6000,
    #                     task_type="CPU",
    #                     thread_count=-1,
    #                     depth=depth,
    #                     l2_leaf_reg=20,
    #                     min_data_in_leaf=1000,
    #                     grow_policy="Lossguide",
    #                     max_leaves=11,
    #                     has_time=True,
    #                     random_seed=4,
    #                     loss_function="MultiClass",
    #                     boosting_type="Plain",
    #                     class_names=current[target].unique(),
    #                 )
    #                 # Fit model
    #                 model.fit(
    #                     X_train,
    #                     y_train,
    #                     eval_set=(X_val, y_val),
    #                     use_best_model=True,
    #                     verbose=1,
    #                     cat_features=cat_features,
    #                     early_stopping_rounds=50,
    #                 )

    #                 if model.best_score_["validation"]["MultiClass"] < best_score:
    #                     best_score = model.best_score_["validation"]["MultiClass"]
    #                     model.save_model(
    #                         f"{models_path}/version_final/model_{airport}_{target}"
    #                     )

    #                 print(
    #                     "@@@@@@@@@@@@@@@",
    #                     airport,
    #                     target,
    #                     eta,
    #                     depth,
    #                     model.best_score_["validation"]["MultiClass"],
    #                 )


if __name__ == "__main__":
    # Read parameters from the command line
    data_path = "data"
    models_path = "\model_scripts\Daniil_scripts\src"
    build_master = True

    # Execute main script
    main(data_path, models_path, build_master)