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
    import os
    import zipfile
    from glob import glob

    import pandas as pd  # type: ignore
    from table_dtype import TableDtype
    from table_generation import generate_table
    from utils import train_test_split

    # the path for root folder
    _ROOT: str = os.path.join(os.path.dirname(__file__), "..", "..")

    # using argparse to parse the argument from command line
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("-s", help="how to save the table")
    parser.add_argument("-a", help="airport")
    parser.add_argument("-m", help="first m rows")
    parser.add_argument("--c", action="store_true", help="combine airport tables")
    args: argparse.Namespace = parser.parse_args()

    # save only a full table - full
    # split the full table into a train table and a validation table and then save these two tables - split
    # I want both (default) - both
    # I want both split and full tables that are saved in a zipped folder - zip
    save_table_as: str = "both" if args.s is None else str(args.s)

    combine_tables = args.c

    # airports need to process
    airports: tuple[str, ...] = ("KATL", "KCLT", "KDEN", "KDFW", "KJFK", "KMEM", "KMIA", "KORD", "KPHX", "KSEA")
    if args.a is not None:
        if combine_tables:
            raise argparse.ArgumentError(None, message="Cannot combine tables when only one airport is being processed")
        airport_selected: str = str(args.a).upper()
        if airport_selected in airports:
            airports = (airport_selected,)
        else:
            raise NameError(f"Unknown airport name {airports}!")

    # the path to the directory where data files are stored
    DATA_DIR: str = os.path.join(_ROOT, "_data")

    airport_tables: list[pd.DataFrame] = []

    for airport in airports:
        print("Processing", airport)

        # extract features for give airport
        table = generate_table(airport, DATA_DIR, -1 if args.m is None else int(args.m))

        # some int features may be missing due to a lack of information
        table = TableDtype.fix_potential_missing_int_features(table)

        # fill the result missing spot with UNK
        table = table.fillna("UNK")

        # table = normalize_str_features(table)

        if combine_tables:
            airport_tables.append(table)
        else:
            # save data
            # full
            if save_table_as == "full" or save_table_as == "both" or save_table_as == "zip":
                table.sort_values(["gufi", "timestamp"]).to_csv(
                    os.path.join(_ROOT, "full_tables", f"{airport}_full.csv"), index=False
                )
            # split
            if save_table_as == "split" or save_table_as == "both" or save_table_as == "zip":
                train_test_split(table, _ROOT, airport)

        print("Finished processing", airport)
        print("------------------------------")

    # if combine is desired, put together big table and save properly according to other arguments
    if combine_tables:
        big_table = pd.concat(airport_tables, ignore_index=True)
        if save_table_as == "full" or save_table_as == "both":
            big_table.sort_values(["gufi", "timestamp"]).to_csv(
                os.path.join(_ROOT, "full_tables", f"ALL_full.csv"), index=False
            )
        if save_table_as == "split" or save_table_as == "both":
            train_test_split(big_table, _ROOT, save=True)

    # zip all generated csv files
    if save_table_as == "zip":
        zip_file_path: str = os.path.join(_ROOT, "all_tables.zip")
        if os.path.exists(zip_file_path):
            os.remove(zip_file_path)
        zip_file: zipfile.ZipFile = zipfile.ZipFile(zip_file_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9)
        for tables_dir in ("train_tables", "validation_tables", "full_tables"):
            for csv_file in glob(os.path.join(_ROOT, tables_dir, "*.csv")):
                zip_file.write(csv_file, os.path.join(tables_dir, os.path.basename(csv_file)))
        zip_file.close()

    print("Done")
