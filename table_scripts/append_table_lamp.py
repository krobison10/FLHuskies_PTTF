#
# Author: Kyler Robison
# Modified by: Daniil Filenko
# 
# This script appends a new column to an existing
# table of all table training data called "train" for a single airport that is hard coded.
#
# To run on compressed data with format specified in README.md, supply a command line
# argument "compressed".
#

import sys
import multiprocessing
import pandas as pd  # type: ignore
import feature_engineering

from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import OrdinalEncoder  # type: ignore
from functools import partial
from pathlib import Path
from tqdm import tqdm


def process_timestamp(now: pd.Timestamp, flights: pd.DataFrame, data_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    # subset table to only contain flights for the table timestamp
    time_filtered_table: pd.DataFrame = flights.loc[flights.timestamp == now].reset_index(drop=True)

    final_table = time_filtered_table.copy()

    # filter features to 30 hours before prediction time to prediction time and save as a copy
    etd: pd.DataFrame = feature_engineering.filter_by_timestamp(data_tables["etd"], now, 30).copy()

    # ----- Minutes Until ETD -----
    # get the latest ETD for each flight
    latest_etd: pd.DataFrame = etd.groupby("gufi").last()

    # get a series containing latest ETDs for each flight, in the same order they appear in flights
    departure_runway_estimated_time: pd.Series = time_filtered_table.merge(
        latest_etd.departure_runway_estimated_time, how="left", on="gufi"
    ).departure_runway_estimated_time

    # add new column to time_filtered_table that represents minutes until pushback
    final_table["minutes_until_etd"] = ((departure_runway_estimated_time - time_filtered_table.timestamp).dt.total_seconds() / 60).astype(int)

    return final_table

def ExtractLampFeatures(table, weather):
    #TODO:Abstract to all of the airports
    # https://github.com/drivendataorg/nasa-airport-config/blob/main/1st%20Place/submission/src/ExtractFeatures.py

    print("Preprocesssing Weather Features")
    table['forecast_timestamp'] = pd.to_datetime(table['forecast_timestamp'])
    table['lightning_prob'] = table['lightning_prob'].map({'L': 0, 'M': 1, 'N': 2, 'H': 3})
    table['precip'] = table['precip'].astype(float)
    table['time_ahead_prediction'] = (table['forecast_timestamp'] - table['timestamp']).dt.total_seconds() / 3600
    table.sort_values(['timestamp', 'time_ahead_prediction'], inplace = True)

    # past_temperatures = table.groupby('timestamp').first().drop(columns = ['forecast_timestamp', 'time_ahead_prediction'])
    # past_temperatures = past_temperatures.rolling('6h').agg({'mean', 'min', 'max'}).reset_index()
    # past_temperatures.columns = ['feat_4_'+c[0]+'_'+c[1]+'_last6h' if c[0]!='timestamp' 
    #                                 else 'timestamp' for c in past_temperatures.columns ]
    # past_temperatures = past_temperatures.set_index('timestamp').resample('15min').ffill().reset_index()
    
    # table_feats = past_temperatures.copy()
    
    # for p in range(1,24):
    #     next_temp = table[(table.time_ahead_prediction <= p) & 
    #                         (table.time_ahead_prediction > p-1)
    #                         ].drop(columns=['forecast_timestamp', 'time_ahead_prediction']).groupby('timestamp').mean().reset_index()
    #     next_temp.columns = ['feat_4_'+c+'_next_'+str(p) if c!='timestamp' else 'timestamp' for c in next_temp.columns]
    #     next_temp = next_temp.set_index('timestamp').resample('15min').ffill().reset_index()
    #     table_feats = table_feats.merge(next_temp, how = 'left', on = 'timestamp')
    
    
    # table_feats['airport'] = airport
    # weather = pd.concat([weather, table_feats])

    # table = table.merge(weather, how='left', on=['timestamp'])
    # master_table = master_table.merge(weather, how='left', on=['airport', 'timestamp'])

    # weather_feats = [c for c in weather.columns if 'feat_4' in c]
    
    # for feat in weather_feats:
    #     table[feat+'_global_min'] = table['timestamp'].map(weather.groupby('timestamp')[feat].min())
    #     table[feat+'_global_mean'] = table['timestamp'].map(weather.groupby('timestamp')[feat].mean())
    #     table[feat+'_global_max'] = table['timestamp'].map(weather.groupby('timestamp')[feat].max())
    #     table[feat+'_global_std'] = table['timestamp'].map(weather.groupby('timestamp')[feat].std())
    
    return table

if __name__ == "__main__":
    ext = ".bz2"

    DATA_DIR = Path("data/")
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

    airport = "KSEA"

    table: pd.DataFrame = pd.read_csv(DATA_DIR / airport / f"prescreened_train_labels_{airport}.csv.bz2", parse_dates=["timestamp"]).sort_values(
            "timestamp"
        )

    # define list of data tables to load and use for each airport
    feature_tables: dict[str, pd.DataFrame] = {
        "etd": pd.read_csv(DATA_DIR / airport / f"{airport}_etd.csv{ext}", parse_dates=["departure_runway_estimated_time", "timestamp"]).sort_values(
            "timestamp"
        ),
        "lamp": pd.read_csv(DATA_DIR / airport / f"{airport}_lamp.csv{ext}",parse_dates=[ "forecast_timestamp", "timestamp"]),
        "mfs": pd.read_csv(DATA_DIR / airport / f"{airport}_mfs.csv{ext}"),
    }

    #Defining necessary variables
    table = table.merge(feature_tables["lamp"][["wind_gust", "cloud_ceiling", "cloud", "lightning_prob", "precip", "wind_speed", "wind_direction", "temperature", "forecast_timestamp", "timestamp"]].fillna("UNK"), how="left", on="timestamp")
    table = table.merge(feature_tables["mfs"][["aircraft_engine_class", "aircraft_type", "major_carrier", "flight_type", "gufi"]].fillna("UNK"), how="left", on="gufi")
    weather = pd.DataFrame()
    table = ExtractLampFeatures(table, weather)
    
    # process all prediction times in parallel
    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        fn = partial(process_timestamp, flights=table, data_tables=feature_tables)
        timestamp_tables: list[pd.DataFrame] = list(tqdm(executor.map(fn, pd.to_datetime(table.timestamp.unique())), total=len(table.timestamp.unique())))

    # concatenate individual prediction times to a single dataframe
    table = pd.concat(timestamp_tables, ignore_index=True)

    # move train label column to the end
    cols = table.columns.tolist()
    cols.remove("minutes_until_pushback")
    cols.append("minutes_until_pushback")
    table = table[cols]

    # save with name "KSEA_train_w_lamp.csv"
    table.to_csv(Path("KSEA_test_w_lamp.csv"), index=False)