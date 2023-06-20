import pandas as pd
from config import *
from collections import defaultdict
from sklearn.model_selection import GroupShuffleSplit
from airline_dataset import *
from torch.utils.data import DataLoader, random_split
import numpy as np
from sklearn.preprocessing import OrdinalEncoder


def load_airports(airport: str) -> dict:
    train_loaders = []
    test_loaders = []
    
    labels = pd.read_csv(f"{P2PATH}/{airport}/phase2_train_labels_{airport}.csv.bz2", parse_dates=["timestamp"])

    for airline in AIRLINES:
        dfs = labels[labels["gufi"].str[:3] == airline]

        etd = pd.read_csv(f"{P2PATH}/{airport}/public/{airport}/{airport}_etd.csv.bz2", parse_dates=["timestamp"])

        latest_etd = etd.sort_values("timestamp").groupby("gufi").last().departure_runway_estimated_time

        #print(latest_etd)

        minutes_until_etd = dfs.merge(
            latest_etd, how="left", on="gufi"
        ).departure_runway_estimated_time

        minutes_until_etd = pd.to_datetime(minutes_until_etd)

        minutes_until_etd = (minutes_until_etd - dfs.timestamp).dt.total_seconds() / 60

        dfs["minutes_until_etd"] = minutes_until_etd

        mfs = pd.read_csv(f"{P2PATH}/{airport}/private/{airport}/{airport}_{airline}_mfs.csv.bz2")
        
        dfs = dfs.merge(mfs, on="gufi")
        #dfs = dfs.dropna()

        dfs["minutes_until_etd"] = dfs["minutes_until_etd"].fillna(0)
        dfs = dfs.fillna("UNK")

        if len(dfs) == 0:
            continue

        try:

            gufi_groups = dfs["gufi"]
            
            splitter = GroupShuffleSplit(test_size=0.2, random_state=42)
            train_indices, test_indices = next(splitter.split(dfs, groups=gufi_groups))
            train_dfs = dfs.iloc[train_indices]
            test_dfs = dfs.iloc[test_indices]

        except ValueError:
            continue

        X_train = train_dfs[features].copy()
        y_train = train_dfs["minutes_until_pushback"].copy()
        X_test = test_dfs[features].copy()
        y_test = test_dfs["minutes_until_pushback"].copy()

        for column in encoded_columns:
            try:
                enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                X_train[column] = enc.fit_transform(X_train[[column]])
                X_test[column] = enc.transform(X_test[[column]])
            except Exception as e:
                print(e)
                print(column)
                exit()

        # for column in X_train.columns:
        #     if column not in encoded_columns:
        #         X_train[column] = (X_train[column]-X_train[column].mean())/X_train[column].std()
        #         X_test[column] = (X_test[column]-X_test[column].mean())/X_test[column].std()

        X_train = X_train.to_numpy(dtype=np.float32, copy=True)
        X_test = X_test.to_numpy(dtype=np.float32, copy=True)
        y_train = y_train.to_numpy(dtype=np.float32, copy=True).reshape(-1, 1)
        y_test = y_test.to_numpy(dtype=np.float32, copy=True).reshape(-1, 1)

        X_train.flags.writeable=True
        y_train.flags.writeable=True
        X_test.flags.writeable=True
        y_test.flags.writeable=True

        #print(X_train)

        train_data = AirlineDataset(X_train, y_train)
        test_data = AirlineDataset(X_test, y_test)

        train_loaders.append(DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True))
        test_loaders.append(DataLoader(test_data, batch_size=BATCH_SIZE))

    return train_loaders, test_loaders