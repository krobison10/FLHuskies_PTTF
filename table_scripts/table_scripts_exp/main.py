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

import os

from table_dtype import TableDtype
from table_generation import generate_table_for
from utils import split

if __name__ == "__main__":
    # output the table as it is - full
    # splitted the full table into a train table and a validation table and then save these two table - splitted
    # I want both - all
    output_df_as: str = "all"

    airports = [
        "KATL",
        "KCLT",
        "KDEN",
        "KDFW",
        "KJFK",
        "KMEM",
        "KMIA",
        "KORD",
        "KPHX",
        "KSEA",
    ]

    label_type: str = "prescreened"

    DATA_DIR: str = os.path.join(os.path.dirname(__file__), "..", "..", "_data")

    for airport in airports:
        print("Start processing:", airport)

        # extract features for give airport
        table = generate_table_for(airport, DATA_DIR)

        # some int features may be missing due to a lack of information
        table = TableDtype.fix_potential_missing_int_features(table)

        # fill the result missing spot with UNK
        table = table.fillna("UNK")

        # adding feature gufi_end_label since it could be useful
        table["gufi_end_label"] = table.apply(lambda x: "TFM" if x.gufi.endswith("TFM") else "TFM_TFDM" if x.gufi.endswith("TFM_TFDM") else "OTHER", axis=1)

        # table = normalize_str_features(table)

        # save data
        if output_df_as == "full" or output_df_as == "all":
            table.sort_values(["gufi", "timestamp"]).to_csv(
                os.path.join(os.path.dirname(__file__), "..", "..", "full_tables", f"main_{airport}_prescreened.csv"), index=False
            )
        if output_df_as == "splitted" or output_df_as == "all":
            split(table, os.path.join(os.path.dirname(__file__), "..", ".."), airport)

        print("Finish processing:", airport)
        print("------------------------------")

    print("Done")
