import pandas as pd
from pathlib import Path

# Converts CSV files into pandas dataframes.

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR.joinpath('generated-data')

def csvToDf(name, data_dir=DATA_DIR, columns: list[str] | None = None) -> pd.DataFrame: 
    file_path = data_dir.joinpath(str(name))
    df = pd.read_csv(file_path, usecols = columns)
    return df
