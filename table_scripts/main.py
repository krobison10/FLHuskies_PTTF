#
# Authors:
# - Kyler Robison
# - Yudong Lin
# - Trevor Tomlin
#
# This script builds a table of training data for a single airport that is hard coded.
#
# It can easily be changed.
#

if __name__ == "__main__":
    import argparse
    import gc
    import os
    import shutil
    import zipfile
    from datetime import datetime
    from glob import glob

    import pandas as pd
    from table_dtype import TableDtype
    from table_generation import generate_table
    from utils import train_test_split

    # the path for root folder
    _ROOT: str = os.path.join(os.path.dirname(__file__), "..")

    # using argparse to parse the argument from command line
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("-s", help="how to save the table")
    parser.add_argument("-a", help="airport")
    parser.add_argument("-m", help="first m rows")
    args: argparse.Namespace = parser.parse_args()

    # save only a full table - full
    # split the full table into a train table and a validation table and then save these two tables - split
    # I want both (default) - both
    # I want both split and full tables that are saved in a zipped folder - zip
    save_table_as: str = "both" if args.s is None else str(args.s)

    # airports need to process
    airports: tuple[str, ...] = ("KATL", "KCLT", "KDEN", "KDFW", "KJFK", "KMEM", "KMIA", "KORD", "KPHX", "KSEA")
    if args.a is not None:
        airport_selected: str = str(args.a).upper()
        if airport_selected in airports:
            airports = (airport_selected,)
        else:
            raise NameError(f"Unknown airport name {airports}!")

    # the path to the directory where data files are stored
    DATA_DIR: str = os.path.join(_ROOT, "_data")

    our_dirs: dict[str, str] = {}

    for airport in airports:
        print("Processing", airport)

        # extract features for give airport
        table: dict[str, pd.DataFrame] = generate_table(airport, DATA_DIR, -1 if args.m is None else int(args.m))

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

        print("Finished processing", airport)
        print("------------------------------")

        gc.collect()

    # put together big table and save properly according to other arguments
    """
    if args.a is None:
        public_master_table: pd.DataFrame = pd.concat(
            [
                pd.read_csv(individual_table, dtype={"precip": str})
                for individual_table in glob(os.path.join(_ROOT, "full_tables", "*", "PUBLIC_full.csv"))
            ],
            ignore_index=True,
        ).sort_values(["gufi", "timestamp"])
        our_dirs = {
            "train_tables": os.path.join(_ROOT, "train_tables"),
            "validation_tables": os.path.join(_ROOT, "validation_tables"),
            "full_tables": os.path.join(_ROOT, "full_tables"),
        }
        if save_table_as == "full" or save_table_as == "both" or save_table_as == "zip":
            public_master_table.to_csv(os.path.join(our_dirs["full_tables"], "PUBLIC_all.csv"), index=False)
        if save_table_as == "split" or save_table_as == "both" or save_table_as == "zip":
            train_test_split(public_master_table, _ROOT, our_dirs, "ALL", "PUBLIC")
        del public_master_table
    """

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

    print("Done")
