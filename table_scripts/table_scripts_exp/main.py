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

    from table_dtype import TableDtype
    from table_generation import generate_table
    from utils import train_test_split

    # using argparse to parse the argument from command line
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("-s", help="how to save the table")
    parser.add_argument("-a", help="airport")
    parser.add_argument("-m", help="first m rows")
    args: argparse.Namespace = parser.parse_args()

    # save the table as it is - full
    # splitted the full table into a train table and a validation table and then save these two table - splitted
    # I want both - all
    save_table_as: str = "all" if args.s is None else str(args.s)

    # airports need to process
    airports: tuple[str, ...] = ("KATL", "KCLT", "KDEN", "KDFW", "KJFK", "KMEM", "KMIA", "KORD", "KPHX", "KSEA")
    if args.a is not None:
        airport_selected: str = str(args.a).upper()
        if airport_selected in airports:
            airports = (airport_selected,)
        else:
            raise NameError(f"Unknown airport name {airports}!")

    DATA_DIR: str = os.path.join(os.path.dirname(__file__), "..", "..", "_data")

    for airport in airports:
        print("Start processing:", airport)

        # extract features for give airport
        table = generate_table(airport, DATA_DIR, -1 if args.m is None else int(args.m))

        # some int features may be missing due to a lack of information
        table = TableDtype.fix_potential_missing_int_features(table)

        # fill the result missing spot with UNK
        table = table.fillna("UNK")

        # table = normalize_str_features(table)

        # save data
        if save_table_as == "full" or save_table_as == "all":
            table.sort_values(["gufi", "timestamp"]).to_csv(
                os.path.join(os.path.dirname(__file__), "..", "..", "full_tables", f"{airport}_full.csv"), index=False
            )
        if save_table_as == "splitted" or save_table_as == "all":
            train_test_split(table, os.path.join(os.path.dirname(__file__), "..", ".."), airport)

        print("Finish processing:", airport)
        print("------------------------------")

        # clear out cache
        gc.collect()

    print("Done")
