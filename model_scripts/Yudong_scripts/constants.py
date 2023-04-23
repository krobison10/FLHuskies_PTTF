from typing import Final

# the column used for target
TARGET_LABEL: str = "minutes_until_pushback"

ALL_AIRPORTS: Final[tuple[str, ...]] = (
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
    # "ALL",
)
