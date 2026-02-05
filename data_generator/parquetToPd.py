import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR.joinpath('data')

# Path to your parquet file
file_path = DATA_DIR.joinpath("ethylene_methane_ds_10hz.parquet")
 
# Load dataset
df = pd.read_parquet(file_path)
print(df)