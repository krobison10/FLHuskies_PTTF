#
# Author: Daniil Filienko (?)
#
# extract potential useful features from configs [latest available runways data]
#

from datetime import datetime
import numpy as np

import pandas as pd  # type: ignore
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
    "KSEA"]

def add_runway_features(_df: pd.DataFrame,raw_data:pd.DataFrame) -> pd.DataFrame:

    features = pd.DataFrame()
    for airport in airports:
        # Filter airport of interest
        current = raw_data[airport].sort_values("timestamp").copy()

        # Add fictitious row in a future timestamp
        enlarge_configs = pd.DataFrame(
            {
                "timestamp": [master_table["timestamp"].max()],
                "airport_config": [current.iloc[-1]["airport_config"]],
                "airport": [airport],
            }
        )
        current = current.append(enlarge_configs)

        # Aggregate at a 15minute time window
        current = current.groupby("timestamp").airport_config.last().reset_index()
        current = (
            current.set_index("timestamp")
            .airport_config.resample("15min")
            .ffill()
            .reset_index()
        )
        current.columns = ["timestamp", "feat_1_cat_airportconfig"]
        current = current[current["feat_1_cat_airportconfig"].isna() == False]

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

        runways = []
        possible_runways = np.unique(raw_data[['departure_runways', 'arrival_runways']].values)

        # making a list of active runways
        for r in possible_runways:
            runway = ''.join(filter(lambda x: x.isdigit(), r))
            runways.append(runway)
        
        # Add binary indicator for each runway indicating if they are active in each direction
        for r in runways:
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