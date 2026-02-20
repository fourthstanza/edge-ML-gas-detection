import pandas as pd
from pathlib import Path

# Converts Apache Parquet files into pandas dataframes.

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR.joinpath('generated-data')

def parquetToDf(name, data_dir=DATA_DIR): 
    file_path = data_dir.joinpath(str(name))
    df = pd.read_parquet(file_path)
    return df
