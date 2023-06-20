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
P2PATH = Path("/home/daniilf/FLHuskies_PTTF/data/files/")

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

# encoded_columns = [
#     "cloud",
#     "lightning_prob",
#     "precip",
#     "gufi_flight_major_carrier",
#     "gufi_flight_destination_airport",
#     "aircraft_engine_class",
#     "aircraft_type",
#     "major_carrier",
#     "flight_type",
# ]

# features = [
#     "gufi_flight_major_carrier",
#     "deps_3hr",
#     "deps_30hr",
#     "arrs_3hr",
#     "arrs_30hr",
#     "deps_taxiing",
#     "arrs_taxiing",
#     "exp_deps_15min",
#     "exp_deps_30min",
#     "standtime_30hr",
#     "dep_taxi_30hr",
#     "arr_taxi_30hr",
#     "minute",
#     "gufi_flight_destination_airport",
#     "month",
#     "day",
#     "hour",
#     "year",
#     "weekday",
#     "minutes_until_etd",
#     "aircraft_engine_class",
#     "aircraft_type",
#     "major_carrier",
#     "flight_type",
#     "temperature",
#     "wind_direction",
#     "wind_speed",
#     "wind_gust",
#     "cloud_ceiling",
#     "visibility",
#     "cloud",
#     "lightning_prob",
#     "precip",
# ]
