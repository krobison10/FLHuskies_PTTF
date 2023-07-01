import pandas as pd
from config import *
from collections import defaultdict
from sklearn.model_selection import GroupShuffleSplit
from airline_dataset import *
from torch.utils.data import DataLoader, random_split
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from functools import partial


def load_airports(airport: str) -> dict:
    train_loaders = []
    test_loaders = []
    
    labels = pd.read_csv(f"{P2PATH}/train_labels_phase2/phase2_train_labels_{airport}.csv.bz2", parse_dates=["timestamp"])

    data = defaultdict()

    encoder = partial(OrdinalEncoder, handle_unknown="use_encoded_value", unknown_value=-1)
    encoders = defaultdict(encoder)

    for airline in AIRLINES:

        try: 
            dfs = pd.read_csv(f"/home/daniilf/FLHuskies_PTTF/full_tables/{airport}/{airline}_full.csv", parse_dates=["timestamp"])
        except FileNotFoundError:
            continue

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

        data[airline] = defaultdict()
        data[airline]["X_train"] = X_train
        data[airline]["X_test"] = X_test
        data[airline]["y_train"] = y_train
        data[airline]["y_test"] = y_test

        for column in encoded_columns:
            try:
                encoders[column].fit(X_train[[column]])
            except Exception as e:
                print(e)
                print(column)
                exit()

    for airline in data.keys():
        X_train = data[airline]["X_train"]
        X_test = data[airline]["X_test"]
        y_train = data[airline]["y_train"]
        y_test = data[airline]["y_test"]

        for column in encoded_columns:
            try:
                X_train[column] = encoders[column].transform(X_train[[column]])
                X_test[column] = encoders[column].transform(X_test[[column]])
            except Exception as e:
                print(e)
                print(column)
                exit()
                
        for column in X_train.columns:
            if column not in encoded_columns:
                X_train[column] = (X_train[column]-X_train[column].mean())/X_train[column].std()
                X_test[column] = (X_test[column]-X_test[column].mean())/X_test[column].std()

        X_train = X_train.to_numpy(dtype=np.float32, copy=True)
        X_test = X_test.to_numpy(dtype=np.float32, copy=True)
        y_train = y_train.to_numpy(dtype=np.float32, copy=True).reshape(-1, 1)
        y_test = y_test.to_numpy(dtype=np.float32, copy=True).reshape(-1, 1)

        X_train.flags.writeable=True
        y_train.flags.writeable=True
        X_test.flags.writeable=True
        y_test.flags.writeable=True

        train_data = AirlineDataset(X_train, y_train)
        test_data = AirlineDataset(X_test, y_test)

        train_loaders.append(DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True))
        test_loaders.append(DataLoader(test_data, batch_size=BATCH_SIZE))

    return train_loaders, test_loaders