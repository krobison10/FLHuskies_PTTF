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

def predict(model, df):
    tensor = torch.from_numpy(df.values).float()
    model.eval()

    with torch.no_grad():
        predictions = model(tensor)

    predictions = predictions.numpy()

    df['minutes_until_pushback'] = predictions

    return df

def load_model(assets_directory):
    """Load all model assets from disk."""
    model = None
    encoder = None
    with open(assets_directory + "/encoders.pickle", 'rb') as fp:
        encoder = pickle.load(fp)
    with open(assets_directory + "/model_5.pt", 'rb') as fp:
        model = torch.load(fp, map_location ='cpu')

    return model, encoder

def encode_df(_df: pd.DataFrame, encoded_columns: list, encoders) -> pd.DataFrame:
    for column in encoded_columns:
        try:
            _df[column] = encoders[column].transform(_df[[column]])
        except Exception as e:
            print(e)
            print(column)
            exit()
    return _df

if __name__ == "__main__":
    import argparse
    import gc
    import os
    import shutil
    import zipfile
    from datetime import datetime
    from glob import glob
    import psutil
    from table_dtype import TableDtype
    from table_generation import generate_table
    from utils import *
    import sys

    p = psutil.Process(os.getpid())
    p.nice(psutil.HIGH_PRIORITY_CLASS)

    # the path for root folder
    _ROOT: str = os.path.join(os.path.dirname(__file__), "..")

    # using argparse to parse the argument from command line
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("-s", help="save")
    parser.add_argument("-a", help="airport")
    parser.add_argument("-m", help="first m rows")
    args: argparse.Namespace = parser.parse_args()

    # save only a full table - full
    # split the full table into a train table and a validation table and then save these two tables - split
    # I want both (default) - both
    # I want both split and full tables that are saved in a zipped folder - zip
    save_table_as: str = "both" if args.s is None else str(args.s)

    # using argparse to parse the argument from command line
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("-a", help="airport")
    parser.add_argument("-m", help="first m rows")
    args: argparse.Namespace = parser.parse_args()

    # airports evaluated for
    airports: tuple[str, ...] = ("KATL", "KCLT", "KDEN", "KDFW", "KJFK", "KMEM", "KMIA", "KORD", "KPHX", "KSEA")
    if args.a is not None:
        airport_selected: str = str(args.a).upper()
        if airport_selected in airports:
            airports = (airport_selected,)
        else:
            raise NameError(f"Unknown airport name {airports}!")

    # the path to the directory where data files are stored
    DATA_DIR: str = os.path.join(_ROOT, "_data")
    ASSETS_DIR: str = os.path.join(_ROOT, "assets")
    TRAIN_DIR: str = os.path.join(_ROOT, "training")

    sys.path.append(TRAIN_DIR)
    from federated import train

    predictions = []
    submission_format = pd.read_csv(f"{ASSETS_DIR}/submission_format.csv", parse_dates=['timestamp'])

    our_dirs: dict[str, str] = {}

    for airport in airports:
        print("Processing", airport)
        # extract features for given airport
        table: dict[str, pd.DataFrame] = generate_table(airport, DATA_DIR, submission_format, -1 if args.m is None else int(args.m))
        
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
        
        # If have not been run before, run the training. 
        if not os.listdir(ASSETS_DIR):
            train()
        model, encoder = load_model(ASSETS_DIR)
        _df = encode_df(table, encoded_columns, encoder)

        #evaluating model
        airport_predictions = predict(model, _df[features])
        predictions.append(airline_predictions)
        del _df
        print("Finished evaluation", airport)
        print("------------------------------")

        gc.collect()

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

    predictions = pd.concat(predictions, axis=0)
    predictions = predictions.loc[submission_format.index]
    predictions.to_csv("submission.csv")
