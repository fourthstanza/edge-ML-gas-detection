import pandas as pd
from pathlib import Path

# Converts fixed width files into pandas dataframes.

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR.joinpath('data')

def fwfToDf(name, data_dir=DATA_DIR, columns: list[str] | None = None) -> pd.DataFrame: 
    file_path = data_dir.joinpath(str(name))
    if name is "ethylene_methane.txt":
        antagonist = "Ethylene"
    if name is "ethylene_methane.txt":
        antagonist = "Methane"
    else:
        antagonist = "Unknown"
    df = pd.read_fwf(DATA_DIR.joinpath(name), skiprows=1, header=None, names=["Time", "Methane ppm", f"{antagonist} ppm", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12", "S13", "S14", "S15", "S16"])
    return df
