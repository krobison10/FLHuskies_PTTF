#
# Authors:
# - Kyler Robison
# - Yudong Lin
# - Trevor Tomlin
# - Daniil Filienko
# This script evaluates loaded model on the validation data.
#
import torch
import pickle
import pandas as pd
import numpy as np
import sys
import os

_ROOT: str = os.path.join(os.path.dirname(__file__), "..")

# the path to the directory where data files are stored
DATA_DIR: str = os.path.join(_ROOT, "data")
ASSETS_DIR: str = os.path.join(_ROOT, "assets")
TRAIN_DIR: str = os.path.join(_ROOT, "training")

sys.path.append(TRAIN_DIR)
from federated import train
from net import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict(model, df):
    numeric_df = df.apply(pd.to_numeric, errors="coerce")

    # Create a new NumPy array with correct numeric data types
    numeric_np_array = numeric_df.values.astype(np.float32)

    # Convert the new NumPy array to a PyTorch tensor
    tensor = torch.from_numpy(numeric_np_array).to(DEVICE)

    # Make predictions using the model
    with torch.no_grad():
        model.eval()  # Set the model to evaluation mode
        predictions = model(tensor)

    # Convert the predictions back to a pandas DataFrame
    df_output = pd.DataFrame(predictions.cpu().numpy(), columns=["minutes_until_pushback"])

    return df_output


def load_model(assets_directory, num):
    """Load all model assets from disk."""
    model = None
    encoder = None
    with open(assets_directory + "/encoders.pickle", "rb") as fp:
        encoder = pickle.load(fp)
    # with open(f"models_new/model_{num}.pt", 'rb') as fp:
    #    model = torch.load(fp)
    with open(assets_directory + "/model.pkl", "rb") as f:
        model = pickle.load(f)
    # model.to('cuda')

    return model, encoder


def encode_df(_df: pd.DataFrame, encoded_columns: list, int_columns: list, encoders) -> pd.DataFrame:
    for column in encoded_columns:
        try:
            _df[column] = encoders[column].transform(_df[[column]])
        except Exception as e:
            print(e)
            print(column)
            print(_df.shape)
    for column in int_columns:
        try:
            _df[column] = _df[column].astype("int")
        except Exception as e:
            print(e)
            print(column)
    return _df


if __name__ == "__main__":
    import argparse
    import gc
    import shutil
    import zipfile
    from datetime import datetime
    from glob import glob
    from table_dtype import TableDtype

    # from table_generation_4 import generate_table
    from utils import *

    # using argparse to parse the argument from command line
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("-t", help="training")
    parser.add_argument("-s", help="save")
    parser.add_argument("-a", help="airport")
    parser.add_argument("-m", help="first m rows")
    parser.add_argument("-n", help="model version")
    args: argparse.Namespace = parser.parse_args()

    # save only a full table - full
    # split the full table into a train table and a validation table and then save these two tables - split
    # I want both (default) - both
    # I want both split and full tables that are saved in a zipped folder - zip
    save_table_as: str = "both" if args.s is None else str(args.s)
    training: bool = False if args.t is None else True
    model_version: int = 5 if args.n is None else int(args.n)

    # airports evaluated for
    airports: tuple[str, ...] = ("KATL", "KCLT", "KDEN", "KDFW", "KJFK", "KMEM", "KMIA", "KORD", "KPHX", "KSEA")
    if args.a is not None:
        airport_selected: str = str(args.a).upper()
        if airport_selected in airports:
            airports = (airport_selected,)
        else:
            raise NameError(f"Unknown airport name {airports}!")

    # If have not been run before, run the training.
    if not os.listdir(ASSETS_DIR):
        training = True

    tables = []
    submission_format = pd.read_csv(f"data/submission_format.csv", parse_dates=["timestamp"])
    # submission_format = pd.read_csv(f'data/', parse_dates = ['timestamp'])
    our_dirs: dict[str, str] = {}
    if training:
        # Run and save training files first
        for airport in airports:
            print("Processing, Training", airport)
            # extract features for given airport
            table: dict[str, pd.DataFrame] = generate_table(
                airport, DATA_DIR, max_rows=-1 if args.m is None else int(args.m)
            )

            # remove old csv
            our_dirs = {
                "train_tables": os.path.join(_ROOT, "train_tables", airport),
                "validation_tables": os.path.join(_ROOT, "validation_tables", airport),
                "full_tables": os.path.join(_ROOT, "full_tables", airport),
            }

            for _out_path in our_dirs.values():
                # remove old csv
                if os.path.exists(_out_path):
                    shutil.rmtree(_out_path)
                # and create new folder
                os.mkdir(_out_path)

            for k in table:
                # some int features may be missing due to a lack of information
                table[k] = TableDtype.fix_potential_missing_int_features(table[k])

                # fix index issue
                table[k].reset_index(drop=True, inplace=True)

                # fill the result missing spot with UNK
                for _col in table[k].select_dtypes(include=["category"]).columns:
                    table[k][_col] = table[k][_col].cat.add_categories("UNK")
                    table[k][_col] = table[k][_col].fillna("UNK").astype(str)

                # fill null
                table[k].fillna("UNK", inplace=True)

                # -- save data ---
                # full
                if save_table_as == "full" or save_table_as == "both" or save_table_as == "zip":
                    table[k].sort_values(["gufi", "timestamp"]).to_csv(
                        os.path.join(our_dirs["full_tables"], f"{k}_full.csv"), index=False
                    )
                # split
                if save_table_as == "split" or save_table_as == "both" or save_table_as == "zip":
                    train_test_split(table[k], _ROOT, our_dirs, airport, k)

            gc.collect()
        # If have not been run before, run the training method.
        print("Training the model")
        train()

    # Validation run
    for airport in airports:
        print("Processing, Inference", airport)
        # extract features for given airport
        table: dict[str, pd.DataFrame] = generate_table(
            airport, DATA_DIR, submission_format, -1 if args.m is None else int(args.m)
        )

        for k in table:
            # some int features may be missing due to a lack of information
            table[k] = TableDtype.fix_potential_missing_int_features(table[k])

            # fix index issue
            table[k].reset_index(drop=True, inplace=True)

            # fill the result missing spot with UNK
            for _col in table[k].select_dtypes(include=["category"]).columns:
                table[k][_col] = table[k][_col].cat.add_categories("UNK")
                table[k][_col] = table[k][_col].fillna("UNK").astype(str)

            # fill null
            table[k].fillna("UNK", inplace=True)
            if save_table_as == "full" or save_table_as == "both" or save_table_as == "zip":
                table[k].sort_values(["gufi", "timestamp"]).to_csv(
                    os.path.join(f"submission_tables/{airport}", f"{k}_full.csv"), index=False
                )
        tables.extend(table.values())

    full_table = pd.concat(tables, axis=0)
    full_table = full_table.drop_duplicates(subset=["gufi", "timestamp", "airport"], keep="last")
    del tables
    full_table = pd.merge(submission_format, full_table, on=["gufi", "timestamp", "airport"], how="inner")
    model, encoder = load_model(ASSETS_DIR, model_version)
    _df = encode_df(full_table, encoded_columns, int_columns, encoder)

    # evaluating the output
    # predictions = predict(model, _df[features])
    predictions = model.predict(_df[features])

    # print(f"Regression tree train error for ALL:", mean_absolute_error(_df["minutes_until_pushback"], predictions))

    output_df = _df[["gufi", "timestamp", "airport"]]
    output_df["minutes_until_pushback"] = predictions  # .values

    del _df

    print("Finished evaluation")
    print("------------------------------")

    output_df = output_df.loc[submission_format.index]
    output_df.to_csv(f"submission.csv")

    # zip all generated csv files
    if save_table_as == "zip":
        current_timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_file_path: str = os.path.join(_ROOT, f"all_tables_{current_timestamp}.zip")
        if os.path.exists(zip_file_path):
            os.remove(zip_file_path)
        zip_file: zipfile.ZipFile = zipfile.ZipFile(zip_file_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6)
        for tables_dir in ("train_tables", "validation_tables", "full_tables"):
            for csv_file in glob(os.path.join(_ROOT, tables_dir, "*", "*.csv")) + glob(
                os.path.join(_ROOT, tables_dir, "*.csv")
            ):
                zip_file.write(csv_file, csv_file[csv_file.index(tables_dir) :])
        zip_file.close()
