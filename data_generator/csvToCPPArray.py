from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR.joinpath('data')

# Change this variable to pick a different dataset to convert
DATA_TO_CONV = '/leak0.csv'

