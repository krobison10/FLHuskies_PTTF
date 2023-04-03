from src.GeneralUtilities import *


def LoadRawData(data_path, airports, start_time, end_time):
    """
    Reads and stores in a dictionary all the data from all the airports
    between start_time and end_time from data_path

    :param str data_path: Parent directory where the data is stored
    :param List[str] airports: List of airports to load the raw data for
    :param str start_time: Timestamp to read from
    :param str end_time: Timestamp to read up to

    :return Dict raw_data: Dictionary mapping data_group string -> dataframe with its contents
    """

    raw_data = {}

    for d in DATA_GROUPS:

        raw_data[d] = {}

        for airport in AIRPORTS:
            current_airport = pd.read_csv(
                DATA_DIR / airport / f"{airport}_{d}.csv.bz2",
                parse_dates=["timestamp"],
            )
            current_airport["airport"] = airport
            current_airport = current_airport[
                (current_airport["timestamp"] >= start_time)
                & (current_airport["timestamp"] <= end_time)
            ]

            if (d == "tfm_estimated_runway_arrival_time") or (
                d == "tbfm_scheduled_runway_arrival_time"
            ):
                current_airport["timestamp"] = current_airport["timestamp"].dt.ceil(
                    "15min"
                )
                current_airport.sort_values("timestamp", inplace=True)
                current_airport = (
                    current_airport.groupby(["gufi", "timestamp"]).last().reset_index()
                )

            raw_data[d][airport] = current_airport

    return raw_data


def CrossJoinDatesAirports(airports, start_time, end_time):
    """
    Creates the cross join between a list of airports and the last 2 days of
    data to create the basis of the master table in which features will be appended

    :param List[str] airports: List of airports to create the master table for
    :param str start_time: Timestamp to create it from
    :param str end_time: Timestamp to create it up to

    :return pd.DataFrame master_table: Cross join of airports and dates
    """

    # airportsDf = pd.DataFrame({"airport": airports})
    # airportsDf["dummy_key"] = 0

    # timestampsDf = pd.DataFrame(
    #     {"timestamp": pd.date_range(start_time, end_time, freq="15min")}
    # )
    # timestampsDf["dummy_key"] = 0

    # master_table = airportsDf.merge(timestampsDf, how="outer", on="dummy_key").drop(
    #     columns="dummy_key"
    # )
    for airport in airports:
        master_table = pd.read_csv(DATA_DIR / airport / f"prescreened_train_labels_{airport}.csv.bz2", parse_dates=["timestamp"]).sort_values(
                "gufi"
            )
        #For the test, return only first 100000 gufis
        master_table = (master_table.iloc[:800000]).sort_values("timestamp")

    #For the test, return only first 10000 gufis
    return master_table

def ExtractAirportconfigFeatures(master_table, raw_data):
    """
    Extracts features based on historical airport configuration of each airport
    these features include number of active departure / arrival runways in
    different rolling windows, past configurations as categorical features...

    :param pd.DataFrame master_table: Cross join of all dates and airports

    :return pd.DataFrame master_table: Airport configuration features
    """

    features = pd.DataFrame()
    for airport in AIRPORTS:
        # Filter airport of interest
        current = raw_data[airport].sort_values("timestamp").copy()

        # Aggregate at a 15minute time window
        current = current.groupby("timestamp").config.last().reset_index()
        current = (
            current.set_index("timestamp")
            .config.resample("15min")
            .ffill()
            .reset_index()
        )
        current.columns = ["timestamp", "feat_1_cat_airportconfig"]
        current = current[current["feat_1_cat_airportconfig"].isna() == False]

        # Add global features related to timestamp
        current["feat_1_cat_hourofday"] = "hour_" + current["timestamp"].dt.hour.astype(
            int
        ).astype(str)
        current["feat_1_cat_dayofweek"] = current["timestamp"].dt.day_name()

        holidays = [str(c)[:10] for c in calendar().holidays(start="2007", end="2024")]
        current["feat_1_isholiday"] = (
            current["timestamp"].apply(lambda x: str(x)[:10] in holidays).astype(int)
        )

        # Indicate number of active runways in each direction
        current["feat_1_active_departurerunways"] = current[
            "feat_1_cat_airportconfig"
        ].apply(
            lambda x: len(
                str(x).replace("_A_", "|").replace("D_", "").split("|")[0].split("_")
            )
        )
        current["feat_1_active_arrivalrunways"] = current[
            "feat_1_cat_airportconfig"
        ].apply(
            lambda x: len(
                str(x).replace("_A_", "|").replace("D_", "").split("|")[1].split("_")
            )
        )

        # Indicate the angle of the configuration
        numbers = current["feat_1_cat_airportconfig"].apply(
            lambda x: [
                int(s)
                for s in x.replace("L", "").replace("R", "").split("_")
                if s.isdigit()
            ]
        )

        current["feat_1_max_directions"] = numbers.apply(
            lambda x: max(x) if len(x) > 0 else 0
        )
        current["feat_1_min_directions"] = numbers.apply(
            lambda x: min(x) if len(x) > 0 else 0
        )
        current["feat_1_unique_directions"] = numbers.apply(
            lambda x: len(set(x)) if len(x) > 0 else 0
        )

        # Rolling variables of active runways for arrival and departures
        for i in [4, 8, 12, 16, 20, 24]:
            current[f"feat_1_active_dep_roll_{i}"] = (
                current[f"feat_1_active_departurerunways"].rolling(i).mean()
            )
            current[f"feat_1_active_arrival_roll_{i}"] = (
                current[f"feat_1_active_arrivalrunways"].rolling(i).mean()
            )
            current[f"feat_1_max_directions_roll_{i}"] = (
                current["feat_1_max_directions"].rolling(i).mean()
            )
            current[f"feat_1_unique_directions_roll_{i}"] = (
                current["feat_1_unique_directions"].rolling(i).mean()
            )

        # Add binary indicator for each runway indicating if they are active in each direction
        for r in RUNWAYS:
            current[f"feat_1_active_dep_{r}"] = (
                current["feat_1_cat_airportconfig"]
                .apply(lambda x: r in x.split("_A_")[0])
                .astype(int)
            )
            current[f"feat_1_active_arrival_{r}"] = (
                current["feat_1_cat_airportconfig"]
                .apply(lambda x: r in x.split("_A_")[1])
                .astype(int)
            )

            current[f"feat_1_active_dep_{r}R"] = (
                current["feat_1_cat_airportconfig"]
                .apply(
                    lambda x: (r in x.split("_A_")[0])
                    and (r + "L" not in x.split("_A_")[0])
                )
                .astype(int)
            )
            current[f"feat_1_active_arrival_{r}R"] = (
                current["feat_1_cat_airportconfig"]
                .apply(
                    lambda x: (r in x.split("_A_")[1])
                    and (r + "L" not in x.split("_A_")[1])
                )
                .astype(int)
            )

            current[f"feat_1_active_dep_{r}L"] = (
                current["feat_1_cat_airportconfig"]
                .apply(
                    lambda x: (r in x.split("_A_")[0])
                    and (r + "R" not in x.split("_A_")[0])
                )
                .astype(int)
            )
            current[f"feat_1_active_arrival_{r}L"] = (
                current["feat_1_cat_airportconfig"]
                .apply(
                    lambda x: (r in x.split("_A_")[1])
                    and (r + "R" not in x.split("_A_")[1])
                )
                .astype(int)
            )

            for i in [4, 8, 12, 16, 20, 24]:
                current[f"feat_1_active_dep_{r}_roll_{i}"] = (
                    current[f"feat_1_active_dep_{r}"].rolling(i).mean()
                )
                current[f"feat_1_active_arrival_{r}_roll_{i}"] = (
                    current[f"feat_1_active_arrival_{r}"].rolling(i).mean()
                )

                current[f"feat_1_active_dep_{r}R_roll_{i}"] = (
                    current[f"feat_1_active_dep_{r}R"].rolling(i).mean()
                )
                current[f"feat_1_active_arrival_{r}R_roll_{i}"] = (
                    current[f"feat_1_active_arrival_{r}R"].rolling(i).mean()
                )

                current[f"feat_1_active_dep_{r}L_roll_{i}"] = (
                    current[f"feat_1_active_dep_{r}L"].rolling(i).mean()
                )
                current[f"feat_1_active_arrival_{r}L_roll_{i}"] = (
                    current[f"feat_1_active_arrival_{r}L"].rolling(i).mean()
                )

        current["feat_1_cat_airconfigother"] = current["feat_1_cat_airportconfig"]
        current.loc[
            (airport + ":" + current["feat_1_cat_airconfigother"]).isin(CLASSES)
            == False,
            "feat_1_cat_airconfigother",
        ] = "other"

        # Add categorical features in 25 lookback period of previous configuration
        for i in range(1, 25):
            current[f"feat_1_cat_previous_config_{i}"] = current[
                "feat_1_cat_airportconfig"
            ].shift(i)

        # Extract features based on number of changes in last periods
        for i in [4, 8, 12, 16, 20, 24]:
            current[f"feat_1_nchanges_last_{i}"] = current[
                [f"feat_1_cat_previous_config_{j}" for j in range(1, 25) if j < i]
            ].nunique(axis=1)

        # Extract features based on the name of the configuration
        current["feat_1_config_length"] = current["feat_1_cat_airportconfig"].apply(
            lambda x: len(str(x))
        )
        current["feat_1_config_nelements"] = current["feat_1_cat_airportconfig"].apply(
            lambda x: len(str(x).split("_"))
        )

        # Insert airport name
        current.insert(0, "airport", airport)

        # Append configuration features with current airport
        features = pd.concat([features, current])

        # Add features of the current airport as global features to the master table
        for i in [4, 8, 12, 16, 20, 24]:
            master_table[f"feat_1_{airport}_dep_roll_{i}"] = master_table[
                "timestamp"
            ].map(
                dict(zip(current["timestamp"], current[f"feat_1_active_dep_roll_{i}"]))
            )
            master_table[f"feat_1_{airport}_arr_roll_{i}"] = master_table[
                "timestamp"
            ].map(
                dict(
                    zip(
                        current["timestamp"], current[f"feat_1_active_arrival_roll_{i}"]
                    )
                )
            )
            master_table[f"feat_1_{airport}_nchanges_{i}"] = master_table[
                "timestamp"
            ].map(dict(zip(current["timestamp"], current[f"feat_1_nchanges_last_{i}"])))

    master_table = master_table.merge(features, how="left", on=["airport", "timestamp"])

    return master_table


def ExtractRunwayArrivalFeatures(master_table, raw_data):
    """
    Extracts features based on past confirmed runway arrival events

    :param pd.Dataframe master_table: Existing feature set at a timestamp-airport level

    :return pd.Dataframe master_table: Master table enlarged with additional features
    """

    features = pd.DataFrame()
    for airport in master_table["airport"].unique():
        current = raw_data[airport].copy()

        current.sort_values("timestamp", inplace=True)
        current["flight_ind1"] = current["gufi"].apply(lambda x: x.split(".")[0])
        current["flight_ind2"] = current["gufi"].apply(lambda x: x.split(".")[1])
        current["indicator"] = 1

        for i in [5, 10, 15, 30, 60, 120, 360]:
            current[f"arrivals_last_{i}min"] = current.rolling(
                f"{i}min", on="timestamp"
            ).indicator.sum()

        current["timestamp"] = current["timestamp"].dt.ceil("15min")
        current = current.groupby("timestamp").agg(
            {
                "indicator": "sum",
                "arrivals_last_5min": "last",
                "arrivals_last_10min": "last",
                "arrivals_last_15min": "last",
                "arrivals_last_30min": "last",
                "arrivals_last_60min": "last",
                "arrivals_last_120min": "last",
                "arrivals_last_360min": "last",
                "flight_ind1": ["last", "nunique"],
                "flight_ind2": ["last", "nunique"],
                "arrival_runway": ["last", "nunique"],
            }
        )
        current.columns = ["feat_2_" + c[0] + "_" + c[1] for c in current.columns]
        current.rename(
            columns={
                "feat_2_flight_ind1_last": "feat_2_cat_flight_ind1_last",
                "feat_2_flight_ind2_last": "feat_2_cat_flight_ind2_last",
                "feat_2_arrival_runway_last": "feat_2_cat_arrival_runway_last",
            },
            inplace=True,
        )
        current.reset_index(inplace=True)
        current["airport"] = airport

        features = pd.concat([features, current])

    master_table = master_table.merge(features, how="left", on=["airport", "timestamp"])

    return master_table

def ExtractMfsFeatures(master_table, airport):
    """
    Extracts mfs features associated with the type of the aircraft

    :param pd.Dataframe master_table: Existing feature set at a timestamp-airport level

    :return pd.Dataframe master_table: Master table enlarged with additional features
    """

    feature_tables: dict[str, pd.DataFrame] = {
    "mfs": pd.read_csv(DATA_DIR / airport / f"{airport}_mfs.csv{EXT}"),
    }

    #Defining necessary variables
    master_table = master_table.merge(feature_tables["mfs"][["aircraft_engine_class", "aircraft_type", "major_carrier", "flight_type", "gufi"]].fillna("UNK"), how="left", on="gufi")

    return master_table

def ExtractRunwayDepartureFeatures(master_table, raw_data):
    """
    Extracts features based on past confirmed runway departure events

    :param pd.Dataframe master_table: Existing feature set at a timestamp-airport level

    :return pd.Dataframe master_table: Master table enlarged with additional features
    """

    features = pd.DataFrame()
    for airport in master_table["airport"].unique():
        current = raw_data[airport].copy()

        current.sort_values("timestamp", inplace=True)
        current["flight_ind1"] = current["gufi"].apply(lambda x: x.split(".")[0])
        current["flight_ind2"] = current["gufi"].apply(lambda x: x.split(".")[1])
        current["indicator"] = 1

        for i in [5, 10, 15, 30, 60, 120, 360]:
            current[f"deps_last_{i}min"] = current.rolling(
                f"{i}min", on="timestamp"
            ).indicator.sum()

        current["timestamp"] = current["timestamp"].dt.ceil("15min")
        current = current.groupby("timestamp").agg(
            {
                "indicator": "sum",
                "deps_last_5min": "last",
                "deps_last_10min": "last",
                "deps_last_15min": "last",
                "deps_last_30min": "last",
                "deps_last_60min": "last",
                "deps_last_120min": "last",
                "deps_last_360min": "last",
                "flight_ind1": ["last", "nunique"],
                "flight_ind2": ["last", "nunique"],
                "departure_runway": ["last", "nunique"],
            }
        )

        current.columns = ["feat_3_" + c[0] + "_" + c[1] for c in current.columns]
        current.rename(
            columns={
                "feat_3_flight_ind1_last": "feat_3_cat_flight_ind1_last",
                "feat_3_flight_ind2_last": "feat_3_cat_flight_ind2_last",
                "feat_3_departure_runway_last": "feat_3_cat_dep_runway_last",
            },
            inplace=True,
        )

        current.reset_index(inplace=True)
        current["airport"] = airport

        features = pd.concat([features, current])

    master_table = master_table.merge(features, how="left", on=["airport", "timestamp"])

    return master_table


def ExtractLampFeatures(master_table, raw_data):
    """
    Extracts features of weather forecasts for each airport and appends it to the
    existing master table

    :param pd.Dataframe master_table: Existing feature set at a timestamp-airport level

    :return pd.Dataframe master_table: Master table enlarged with additional features
    """

    weather = pd.DataFrame()
    for airport in master_table["airport"].unique():
        current = raw_data[airport].copy()

        current["forecast_timestamp"] = pd.to_datetime(current["forecast_timestamp"])
        current["lightning_prob"] = current["lightning_prob"].map(
            {"L": 0, "M": 1, "N": 2, "H": 3}
        )
        current["cloud"] = (
            current["cloud"]
            .map({"OV": 4, "BK": 3, "CL": 0, "FW": 1, "SC": 2})
            .fillna(3)
        )
        current["precip"] = current["precip"].astype(float)
        current["time_ahead_prediction"] = (
            current["forecast_timestamp"] - current["timestamp"]
        ).dt.total_seconds() / 3600
        current.sort_values(["timestamp", "time_ahead_prediction"], inplace=True)

        past_temperatures = (
            current.groupby("timestamp")
            .first()
            .drop(columns=["forecast_timestamp", "time_ahead_prediction"])
        )
        past_temperatures = (
            past_temperatures.rolling("6h").agg({"mean", "min", "max"}).reset_index()
        )
        past_temperatures.columns = [
            "feat_4_" + c[0] + "_" + c[1] + "_last6h"
            if c[0] != "timestamp"
            else "timestamp"
            for c in past_temperatures.columns
        ]
        past_temperatures = (
            past_temperatures.set_index("timestamp")
            .resample("15min")
            .ffill()
            .reset_index()
        )

        current_feats = past_temperatures.copy()

        for p in range(1, 24):
            next_temp = (
                current[
                    (current.time_ahead_prediction <= p)
                    & (current.time_ahead_prediction > p - 1)
                ]
                .drop(columns=["forecast_timestamp", "time_ahead_prediction"])
                .groupby("timestamp")
                .mean()
                .reset_index()
            )
            next_temp.columns = [
                "feat_4_" + c + "_next_" + str(p) if c != "timestamp" else "timestamp"
                for c in next_temp.columns
            ]
            next_temp = (
                next_temp.set_index("timestamp").resample("15min").ffill().reset_index()
            )
            current_feats = current_feats.merge(next_temp, how="left", on="timestamp")

        current_feats["airport"] = airport

        weather = pd.concat([weather, current_feats])

    master_table = master_table.merge(weather, how="left", on=["airport", "timestamp"])

    # Add global weather features
    weather_feats = [c for c in weather.columns if "feat_4" in c]
    for feat in weather_feats:
        master_table[feat + "_global_min"] = master_table["timestamp"].map(
            weather.groupby("timestamp")[feat].min()
        )
        master_table[feat + "_global_mean"] = master_table["timestamp"].map(
            weather.groupby("timestamp")[feat].mean()
        )
        master_table[feat + "_global_max"] = master_table["timestamp"].map(
            weather.groupby("timestamp")[feat].max()
        )
        master_table[feat + "_global_std"] = master_table["timestamp"].map(
            weather.groupby("timestamp")[feat].std()
        )

    return master_table


def ExtractETDFeatures(master_table, raw_data):
    """
    Extracts estimated time of departure features and appends it to the existing dataframe

    :param pd.DataFrame master_table: Existing feature set at a timestamp-airport level

    :return pd.Dataframe master_table: Master table enlarged with additional features
    """

    etd_features = pd.DataFrame()
    for airport in master_table["airport"].unique():
        etd = raw_data[airport].copy()
        etd["timestamp"] = etd.timestamp.dt.ceil("15min")
        etd["departure_runway_estimated_time"] = pd.to_datetime(
            etd["departure_runway_estimated_time"]
        )
        etd = etd[etd["timestamp"] < etd["departure_runway_estimated_time"]]
        etd["flight_no"] = etd.gufi.apply(lambda x: str(x).split(".")[0])
        etd["flight_arr"] = etd.gufi.apply(lambda x: str(x).split(".")[2])

        complete_etd = etd.copy()
        for i in range(1, 4 * 25):
            current = etd.copy()
            current["timestamp"] = current["timestamp"] + pd.Timedelta(f"{i * 15}min")
            current = current[
                current["timestamp"] < current["departure_runway_estimated_time"]
            ]
            complete_etd = pd.concat([complete_etd, current])

        complete_etd["time_ahead"] = (
            complete_etd["departure_runway_estimated_time"] - complete_etd["timestamp"]
        ).dt.total_seconds()
        complete_etd = complete_etd.groupby(["gufi", "timestamp"]).first().reset_index()

        for i in [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360, 720]:
            complete_etd[f"estdep_next_{i}min"] = (
                complete_etd["time_ahead"] < i * 60
            ).astype(int)
        complete_etd.sort_values("time_ahead", inplace=True)

        etd_aggregation = (
            complete_etd.groupby("timestamp")
            .agg(
                {
                    "gufi": "count",
                    "estdep_next_30min": "sum",
                    "estdep_next_60min": "sum",
                    "estdep_next_90min": "sum",
                    "estdep_next_120min": "sum",
                    "estdep_next_150min": "sum",
                    "estdep_next_180min": "sum",
                    "estdep_next_210min": "sum",
                    "estdep_next_240min": "sum",
                    "estdep_next_270min": "sum",
                    "estdep_next_300min": "sum",
                    "estdep_next_330min": "sum",
                    "estdep_next_360min": "sum",
                    "estdep_next_720min": "sum",
                }
            )
            .reset_index()
        )

        etd_aggregation.columns = [
            "feat_5_" + c if c != "timestamp" else c for c in etd_aggregation.columns
        ]
        etd_aggregation["airport"] = airport
        etd_features = pd.concat([etd_features, etd_aggregation])

    master_table = master_table.merge(
        etd_features, how="left", on=["airport", "timestamp"]
    )

    return master_table


def ExtractERAFeatures(master_table, raw_data, column="estimated_runway_arrival_time"):
    """
    Extracts estimated runway arrival features and appends them to existing master table

    :param pd.DataFrame master_table: Existing feature set at a timestamp-airport level

    :return pd.Dataframe master_table: Master table enlarged with additional features
    """

    era_features = pd.DataFrame()
    for airport in master_table["airport"].unique():
        era = raw_data[airport].copy()
        era[column] = pd.to_datetime(era[column])
        era = era[era["timestamp"] < era[column]]

        complete_era = era.copy()
        for i in range(1, 4 * 25):
            current = era.copy()
            current["timestamp"] = current["timestamp"] + pd.Timedelta(f"{i * 15}min")
            current = current[current["timestamp"] < current[column]]
            complete_era = pd.concat([complete_era, current])

        complete_era["time_ahead"] = (
            complete_era[column] - complete_era["timestamp"]
        ).dt.total_seconds()
        complete_era = complete_era.groupby(["gufi", "timestamp"]).first().reset_index()

        for i in [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360]:
            complete_era[f"next_{i}min"] = (complete_era["time_ahead"] < i * 60).astype(
                int
            )
        complete_era.sort_values("time_ahead", inplace=True)

        era_aggregation = (
            complete_era.groupby("timestamp")
            .agg(
                {
                    "gufi": "count",
                    "next_30min": "sum",
                    "next_60min": "sum",
                    "next_90min": "sum",
                    "next_120min": "sum",
                    "next_150min": "sum",
                    "next_180min": "sum",
                    "next_210min": "sum",
                    "next_240min": "sum",
                    "next_270min": "sum",
                    "next_300min": "sum",
                    "next_330min": "sum",
                    "next_360min": "sum",
                }
            )
            .reset_index()
        )

        prefix = "feat_6_" if column == "estimated_runway_arrival_time" else "feat_9_"

        era_aggregation.columns = [
            prefix + c if c != "timestamp" else c for c in era_aggregation.columns
        ]
        era_aggregation["airport"] = airport
        era_features = pd.concat([era_features, era_aggregation])

    master_table = master_table.merge(
        era_features, how="left", on=["airport", "timestamp"]
    )

    return master_table


def ExtractGufiTimestampFeatures(master_table, raw_data, group):
    """
    Extracts features from a table containing Gufi and Timestamp information
    Serves for katl_mfs_runway_arrival_time, katl_mfs_runway_departure_time,
    katl_mfs_stand_arrival_time and katl_mfs_stand_departure_time

    :param pd.DataFrame master_table: Existing feature set at a timestamp-airport level
    :param str group: Group indicating which df we are using to rename the features

    :return pd.Dataframe master_table: Master table enlarged with additional features
    """

    gufi_features = pd.DataFrame()

    for airport in master_table["airport"].unique():
        current = raw_data[airport].copy()

        current = current.sort_values("timestamp").set_index("timestamp")
        current.index.names = ["date"]
        current["timestamp"] = current.index.ceil("15min")

        for intervals in [
            "15min",
            "30min",
            "45min",
            "60min",
            "90min",
            "120min",
            "240min",
            "360min",
            "480min",
            "720min",
            "24h",
        ]:
            current[f"feat_{group}_count_{intervals}"] = current.rolling(
                intervals
            ).gufi.count()

        counts = [c for c in current.columns if "count" in c]

        current = current.groupby("timestamp")[counts].max().reset_index()
        current["airport"] = airport
        gufi_features = pd.concat([gufi_features, current])

    master_table = master_table.merge(
        gufi_features, how="left", on=["airport", "timestamp"]
    )

    return master_table


def AddTargets(df):
    """
    Includes target variables to the master table feature as shifted configurations at different
    time horizons to be predicted in the multiclass routine

    :param pd.Dataframe df: Dataframe containing the master table

    :return pd.Dataframe df: Master table enlarged with all the targets
    """

    df = df.sort_values(["airport", "timestamp"]).reset_index(drop=True)

    for look in LOOKAHEADS:
        df[f"target_{look}"] = df.groupby("airport").feat_1_cat_airportconfig.shift(
            -int(look / 15)
        )
        df[f"target_{look}"] = df["airport"] + ":" + df[f"target_{look}"]
        df.loc[df[f"target_{look}"].isin(CLASSES) == False, f"target_{look}"] = (
            df["airport"] + ":other"
        )

    return df


def Adjust(df):
    """
    Adjusts the master table with all the features to prevent errors in prediction time
    as a consequence of missing data in some of the datablocks we could have nullable
    fields in the master table

    :param pd.Dataframe df: Dataframe containing the master table

    :return pd.Dataframe df: Master table adjusted
    """

    for col in df.columns:
        if "feat" in col:
            df[col] = df[col].ffill()

            if "_cat_" in col:
                df[col] = df[col].fillna("None")
            else:
                df[col] = df[col].fillna(0)

    return df