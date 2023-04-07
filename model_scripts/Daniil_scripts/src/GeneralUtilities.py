import os
import sys
import time
import typer
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from tqdm import tqdm as tqdm
from datetime import datetime
from catboost import Pool, CatBoostClassifier
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import gc

import warnings

warnings.filterwarnings("ignore")


# Define global variables

LOOKAHEADS = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360]

AIRPORTS = [
    # "KATL",
    # "KCLT",
    # "KDEN",
    # "KDFW",
    # "KJFK",
    # "KMEM",
    # "KMIA",
    # "KORD",
    # "KPHX",
    "KSEA"
]

RUNWAYS = [
    "34",
    "35",
    "31",
    "36",
    "4",
    "18",
    "27",
    "22",
    "7",
    "8",
    "17",
    "25",
    "16",
    "26",
    "28",
    "30",
    "10",
    "9",
    "13",
    "12",
]

DATA_GROUPS = [
    "config",
    "etd",
    "lamp",
    "first_position",
    # "mfs",
    # "tbfm",
]
EXT = ".bz2"

DATA_DIR = Path("_data")

# Auxiliary function to handle the errors


def return_on_failure(value):
    def decorate(f):
        def applicator(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except:
                print("Error")
            return value

        return applicator

    return decorate


# All classes that appeared in the training set

CLASSES = [
    "kden:D_34R_A_34R_35L_35R",
    "kdfw:D_31L_35L_36R_A_31R_35C_35R_36R",
    "kclt:D_36R_A_36R",
    "kjfk:D_31L_4L_A_4L_4R",
    "kmem:D_18C_18R_A_18C_18R_27",
    "kmem:D_18C_18L_A_18L_27",
    "kmem:D_36L_36R_A_36L_36R",
    "kden:D_34R_A_35L",
    "ksea:D_34R_A_34C_34R",
    "kclt:D_18L_A_18L_18R",
    "kord:D_22L_28C_A_28C",
    "kphx:D_7R_8_A_7R_8",
    "kdfw:D_35L_36R_A_31R_35C_35R",
    "kmem:D_18C_18L_18R_27_A_18C_18L_18R_27",
    "kdfw:D_17R_A_17C_17R",
    "kden:D_17L_25_A_16L_16R_17R",
    "kden:D_34L_8_A_35L_35R",
    "katl:D_26L_27L_A_26R_27L_28",
    "kmem:D_27_36L_36R_A_27_36L_36R",
    "kmia:D_26L_26R_27_A_26L_27",
    "kmia:D_8L_8R_A_8R",
    "kord:D_28C_A_27L",
    "kden:D_17L_17R_8_A_17R",
    "kord:D_9C_A_10C",
    "kmia:D_26L_26R_27_30_A_26L_27_30",
    "kden:D_17R_8_A_16L_16R_17R",
    "kden:D_25_34L_8_A_35L_35R",
    "kphx:D_25L_26_A_25L_26",
    "kmia:D_26L_26R_27_A_26L_26R_27",
    "kphx:D_7L_8_A_7L_8",
    "kord:D_10L_9R_A_10C_10R_9L",
    "kden:D_17L_17R_A_16L_16R_17R",
    "kdfw:D_35L_36R_A_35C_35R_36L",
    "kmem:D_18L_18R_A_18L_18R",
    "kmia:D_26L_26R_27_A_26L_27_30",
    "katl:D_26L_27R_A_26R_27L_28",
    "kmem:D_18C_18L_18R_A_18C_18L_18R",
    "kord:D_28C_A_28C",
    "kmem:D_27_36C_A_27_36R",
    "kmem:D_36C_36L_36R_9_A_36L_36R_9",
    "kden:D_17L_17R_8_A_16L_16R_17R",
    "kdfw:D_35C_36R_A_35C_36R",
    "kmem:D_18C_18L_18R_A_18C_18L_18R_27",
    "kden:D_34L_34R_8_A_34R_35L_35R",
    "kdfw:D_35L_36R_A_31R_35C_35R_36R",
    "kjfk:D_4L_A_4L",
    "kmia:D_26L_26R_27_A_26L_26R_27_30",
    "kord:D_27C_28R_A_28R",
    "katl:D_26L_28_A_26R_27L_28",
    "katl:D_26R_27R_A_26R_27L_28",
    "katl:D_26L_27R_A_26L_27R_28",
    "kord:D_9C_A_9C",
    "kphx:D_7L_7R_A_7L_7R",
    "katl:D_8R_9L_A_8R_9L",
    "kclt:D_18C_A_18C",
    "kmem:D_36C_A_36L_36R",
    "kden:D_25_34L_34R_A_34R_35L_35R",
    "kmem:other",
    "kmem:D_36C_36L_A_36C_36L",
    "ksea:D_16L_A_16L_16R",
    "kdfw:D_17R_18L_A_17C_17L_18L",
    "kord:D_10L_9R_A_10C_9L_9R",
    "kphx:D_25L_25R_A_25L_25R",
    "kdfw:D_17R_18L_A_13R_17C_17L",
    "kden:D_17R_A_16R_17R",
    "ksea:D_16C_A_16C_16R",
    "katl:D_26L_27R_28_A_26R_27L_28",
    "kmem:D_36C_A_27_36L_36R",
    "kmem:D_27_36C_36L_36R_A_27_36L_36R",
    "kden:D_34L_8_A_35L_35R_7",
    "katl:other",
    "kclt:D_36C_36R_A_36C_36R",
    "kmia:D_8L_8R_9_A_8R_9",
    "kden:D_17R_A_17R",
    "kord:D_27C_A_27C",
    "kdfw:D_35L_36R_A_31R_35C_35R_36L",
    "kmem:D_18C_A_18L_18R",
    "katl:D_10_8R_A_10_8R",
    "kjfk:D_31L_A_31L",
    "kord:D_4L_A_4R",
    "ksea:D_34C_A_34C_34L",
    "kden:D_17R_A_16L_16R_17R",
    "katl:D_8L_9L_A_10_8L_9R",
    "kden:D_34L_34R_A_34R_35L_35R",
    "kdfw:D_35L_36R_A_35C_35L_36L_36R",
    "kden:D_17R_8_A_16L_17R",
    "kmia:D_8L_8R_9_A_12_8L_8R_9",
    "kord:D_10L_4L_A_10C",
    "kphx:other",
    "kord:D_28R_A_27L_28R",
    "kphx:D_7L_8_A_7R_8",
    "kdfw:D_17R_18L_A_17C_18R",
    "kmia:D_12_8R_9_A_12_8R_9",
    "kmia:D_12_8L_8R_9_A_12_8R_9",
    "kmia:D_8L_9_A_9",
    "kden:D_17L_8_A_16L_16R_17R",
    "katl:D_26L_27R_A_27L_28",
    "kden:D_34L_8_A_34R_35L_35R",
    "katl:D_8R_9L_A_8L_9R",
    "kphx:D_25R_26_A_25R_26",
    "kmem:D_27_36C_36L_36R_A_27_36C_36L_36R",
    "kmia:D_12_8L_8R_9_A_12_9",
    "kmia:D_12_8L_8R_9_A_12_8L_8R_9",
    "kdfw:D_17R_18L_A_13R_17C_17L_18R",
    "ksea:D_16L_A_16C",
    "katl:D_8R_9L_A_10_8R_9R",
    "kmem:D_18C_18L_18R_A_18C",
    "kden:D_17L_17R_25_8_A_16L_16R_17R",
    "kmem:D_36C_36L_36R_A_36C",
    "katl:D_26L_28_A_26L_28",
    "kclt:D_36C_36R_A_36C_36L_36R",
    "kden:D_34L_A_35L",
    "kdfw:D_31L_35L_A_31R_35C_35R_36R",
    "kmia:D_12_9_A_12_9",
    "kord:D_10L_9C_A_10C_10R",
    "kden:D_17L_25_8_A_16R_17R",
    "kclt:other",
    "kmia:D_26L_27_A_26L",
    "ksea:D_16L_A_16C_16L",
    "katl:D_10_8R_9L_A_10_8L_9R",
    "kjfk:D_13R_A_13L_22L",
    "katl:D_9L_A_9R",
    "kjfk:D_4L_A_4L_4R",
    "kden:other",
    "kmem:D_18C_18R_A_18C_18R",
    "kdfw:D_35L_A_35C",
    "kclt:D_18C_A_18C_18R",
    "kord:D_10L_9R_A_10C",
    "kden:D_25_34L_A_35L_35R",
    "katl:D_26L_28_A_26R_28",
    "katl:D_8R_9R_A_10_8L_9R",
    "ksea:other",
    "kden:D_17R_8_A_17R",
    "kden:D_34L_A_34R_35L_35R",
    "kord:D_28R_A_28R",
    "kmia:D_12_8L_8R_9_A_8L_8R_9",
    "ksea:D_16L_A_16C_16R",
    "kmem:D_18C_18L_18R_27_A_18L_18R_27",
    "kmia:D_26L_26R_27_A_26L_26R",
    "kord:D_10L_9C_A_10C_10R_9L",
    "kord:D_22L_28R_A_27C_27R_28C",
    "kphx:D_7L_7R_8_A_7L_7R_8",
    "kdfw:D_17R_18L_A_17R_18L",
    "kjfk:D_22R_31L_A_22L_22R",
    "kord:D_10L_A_9C",
    "kdfw:other",
    "kord:D_22L_28R_A_27C_28C",
    "kord:D_10L_A_10C_10R_9L",
    "kphx:D_25R_26_A_25L_26",
    "kden:D_17L_8_A_17R",
    "kden:D_17R_A_16L_17R",
    "kmia:D_8L_9_A_8L_9",
    "kmem:D_18C_18L_18R_A_18L_18R_27",
    "kord:D_22L_28R_A_27R_28C",
    "katl:D_26L_27R_A_26R_27L",
    "kmia:D_8L_8R_9_A_8L_8R_9",
    "kden:D_17L_8_A_16R_17R",
    "katl:D_26L_27R_A_26R_27R_28",
    "katl:D_26L_27R_A_26R_28",
    "kmem:D_36C_36L_36R_A_27_36L_36R",
    "kphx:D_7L_A_7R_8",
    "kphx:D_7L_A_7L_7R",
    "kclt:D_36C_A_36C",
    "kclt:D_36R_A_36L_36R",
    "kmia:D_8L_8R_A_8L_8R",
    "kjfk:D_31L_A_31L_31R",
    "kden:D_34L_A_35L_35R",
    "kden:D_17L_17R_8_A_16R_17R",
    "kord:D_10L_A_10L",
    "kclt:D_18C_18L_A_18C_18L_18R",
    "kord:D_22L_28R_A_28C",
    "kden:D_25_8_A_16R_35L_35R",
    "kden:D_17R_8_A_16R_17R",
    "ksea:D_16C_A_16L_16R",
    "kmia:D_12_8L_8R_9_A_12_8R",
    "kjfk:D_13R_A_13L",
    "kden:D_17L_25_8_A_16L_16R_17R",
    "kord:D_22L_28R_A_27C",
    "kmem:D_36C_36L_36R_A_36L_36R",
    "kmia:D_8R_9_A_8R_9",
    "kord:D_10L_A_10C",
    "kord:other",
    "kden:D_25_34L_34R_A_26_35L_35R",
    "kdfw:D_17R_18L_A_17C_17R_18L",
    "kden:D_25_34L_34R_8_A_34R_35L_35R",
    "kdfw:D_17R_A_17C",
    "kord:D_28R_A_27C",
    "kden:D_34L_34R_A_35L_35R",
    "kord:D_22L_A_22R",
    "kmia:D_26L_26R_A_26L",
    "kdfw:D_17R_18L_A_17C_17L_18R",
    "kdfw:D_31L_35L_36R_A_31R_35C_35R",
    "kphx:D_7L_A_7R",
    "katl:D_10_8L_A_10_8L",
    "kord:D_28R_A_28C",
    "kmia:D_26L_27_A_26L_27",
    "kjfk:D_22R_A_22L",
    "kmem:D_36C_36L_36R_A_36C_36L_36R",
    "kdfw:D_17R_18L_A_17C_18L",
    "kdfw:D_17R_18L_A_13R_17C_17L_18L",
    "kmem:D_18L_18R_A_18L_18R_27",
    "ksea:D_34R_A_34L_34R",
    "kjfk:D_22R_A_22L_22R",
    "ksea:D_34R_A_34C",
    "kdfw:D_35L_A_35L",
    "kclt:D_18C_18L_A_18C_18L",
    "kjfk:D_13R_A_22L",
    "kord:D_22L_28R_A_28R",
    "kjfk:D_31L_A_31R",
    "kmia:D_26L_26R_A_26L_26R",
    "kclt:D_36C_A_36C_36L",
    "kmem:D_18C_18L_18R_A_18L_18R",
    "kdfw:D_17C_A_17C",
    "kord:D_10L_22L_9C_A_10C_10R_9L",
    "ksea:D_16L_A_16R",
    "katl:D_26R_28_A_26R_28",
    "kden:D_25_34L_8_A_34R_35L_35R",
    "kden:D_25_34L_A_26_35L_35R",
    "kord:D_10L_9R_A_10C_9L",
    "katl:D_8R_9L_A_10_8L_9R",
    "kmem:D_18C_18L_18R_27_A_27",
    "kdfw:D_17C_18L_A_17C_18L",
    "kord:D_10L_9C_A_10C_9L",
    "kmem:D_18C_A_18L_18R_27",
    "kdfw:D_17R_18L_A_17R_18L_18R",
    "kphx:D_25R_A_25R",
    "kphx:D_25R_A_25R_26",
    "kdfw:D_35L_36R_A_35C_35L_36R",
    "kphx:D_25R_A_25L_26",
    "kord:D_10C_A_10C",
    "kdfw:D_18L_A_18L",
    "kphx:D_25R_A_25L_25R",
    "kmia:other",
    "katl:D_26L_27R_A_26L_27R",
    "kden:D_25_34L_A_34R_35L_35R",
    "kmia:D_12_8L_8R_A_12_8L_8R",
    "kord:D_10L_9C_A_10C",
    "katl:D_8R_9L_A_10_9R",
    "kphx:D_7L_A_7L",
    "kden:D_17L_8_A_16R_17R_7",
    "kord:D_22L_28R_A_27C_27R",
    "kmem:D_36C_A_36L_36R_9",
    "kord:D_22L_28R_A_27L_27R_28C",
    "kmia:D_26L_26R_27_30_A_26L_26R_27_30",
    "kdfw:D_17R_18L_A_17C_17R_18L_18R",
    "kclt:D_18L_A_18L",
    "kjfk:D_31L_31R_A_31L_31R",
    "kjfk:other",
    "katl:D_26L_27R_A_26L_27L_28",
    "kmem:D_27_36C_36L_A_27_36C_36L",
    "kden:D_17L_17R_A_16R_17R",
    "kord:D_22L_28R_A_27L_28C",
    "kdfw:D_35L_36R_A_35C_36R",
    "kdfw:D_35L_36R_A_35C_35R_36R",
]
