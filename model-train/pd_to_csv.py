import pandas as pd
from pathlib import Path

# Converts CSV files into pandas dataframes.

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR.joinpath('data')

def dfToCsv(df: pd.DataFrame, name, data_dir=DATA_DIR, columns: list[str] | None = None): 
    file_path = data_dir.joinpath(str(name))
    df.to_csv(file_path, columns = columns)
    return
