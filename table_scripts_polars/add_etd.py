#
# Author: Kyler Robison
#
# calculate and add etd information to the data frame
#

import polars as pl


# calculate etd
def add_etd(flights_selected: pl.DataFrame, latest_etd: pl.DataFrame) -> pl.DataFrame:
    # get a series containing latest ETDs for each flight, in the same order they appear in flights
    flights_selected = flights_selected.join(latest_etd.drop("timestamp"), how="left", on="gufi")

    # add new column to flights_selected that represents minutes until pushback
    flights_selected = flights_selected.with_columns(
        (pl.col("departure_runway_estimated_time") - pl.col("timestamp")).alias("minutes_until_etd")
    )
    flights_selected.drop_in_place("departure_runway_estimated_time")

    flights_selected = flights_selected.with_columns(
        pl.col("minutes_until_etd").apply(lambda x: x.total_seconds() // 60)
    )

    return flights_selected
