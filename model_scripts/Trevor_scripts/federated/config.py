import torch
from pathlib import Path

DATA_DIRECTORY = Path("data_apr16/tables/full_tables/")
ROOT1 = Path("/home/ydlin/FLHuskies_PTTF/")
ROOT = Path("/home/ttomlin/FLHuskies_PTTF")
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
client_resources = None
if DEVICE.type == "cuda":
    client_resources = {"num_gpus": 1}


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


AIRLINES = [
    "AAL",
    "AJT",
    "ASA",
    "ASH",
    "AWI",
    "DAL",
    "EDV",
    "EJA",
    "ENY",
    "FDX",
    "FFT",
    "GJS",
    "GTI",
    "JBU",
    "JIA",
    "NKS",
    "PDT",
    "QXE",
    "RPA",
    "SKW",
    "SWA",
    "SWQ",
    "TPA",
    "UAL",
    "UPS",
]


features = [
    "minutes_until_etd",
    "aircraft_engine_class",
    "aircraft_type",
    "flight_type",
]


encoded_columns = [
    "aircraft_engine_class",
    "aircraft_type",
    "flight_type",
]
