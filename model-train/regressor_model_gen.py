###
# This script is used to generate a regression model for predicting the concentration of a gas (e.g., methane) based on time-series data. 
# It reads a dataset in Apache Parquet format, processes it, and trains a regression model using Keras with TensorFlow as the backend.
# Ensure that the data is complete and has no NaN or Null values, as no handling is done for these cases and the output will be undefined.
###

# ------------- SET DATASET FEATURES HERE -------------- #

FILE_NAME: str = 'ethylene_methane_ds_10hz.parquet'    # Set name of dataset. Ensure it is placed in generated-data (shape of the ethylene-methane dataset is [time_s, methane_ppm, ethylene_ppm, feature1, ..., feature16])
PARQUET: bool = True                                   # Set only the filetype which corresponds to the dataset you are using as true.
CSV: bool = False

INPUTS: int = 16                                       # Number of features to be used for training the model
X_COLUMNS: list = list(range(3, 3+INPUTS))             # Column(s) to be used as features 
Y_COLUMNS: list | int = 1                              # Column(s) to be used as the target variable. Only use a list if there are multiple target variables.
T_COLUMN:  int = 0                                     # Column representing time (in seconds)
FREQUENCY: int = 10                                    # Sampling frequency of the dataset (in Hz)

# -------- SET DATASET HANDLING OPTIONS HERE ----------- #

WINDOW_SIZE: int = 200                                 # Size of the sliding window for creating sequences of data for training the model (in seconds)
SEED: int = 42                                         # Random seed for reproducibility of the train-test-validation split

# ------------------------------------------------------ #

import numpy as np
import random
import tensorflow as tf # Use Keras API from TensorFlow

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR.joinpath('generated-data')

from parquet_to_pd import parquetToDf
from csv_to_pd import csvToDf
from data_split import DataSplit

def create_data_split(data: np.ndarray, split: tuple[float, float, float] | None = None, seed: int | None = None) -> DataSplit:

    winlength = WINDOW_SIZE * FREQUENCY

    X = data[:, X_COLUMNS]
    y = data[:, Y_COLUMNS]
    t = data[:, T_COLUMN]

    splt = DataSplit(X, y, winlength, split=split, time = t, seed = seed)

    return splt

def main(): 

    if PARQUET:
        df = parquetToDf(FILE_NAME)
    elif CSV:
        df = csvToDf(FILE_NAME)
    else:
        raise ValueError("No filetype set to true. Set the filetype corresponding to the dataset being used to true.")

    data = df.to_numpy()

    data_split = create_data_split(data, seed = SEED)

    


if __name__=="__main__": # run main fct if this file is executed as a script
    main()