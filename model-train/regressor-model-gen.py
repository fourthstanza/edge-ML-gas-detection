###
# This script is used to generate a regression model for predicting the concentration of a gas (e.g., methane) based on time-series data. 
# It reads a dataset in Apache Parquet format, processes it, and trains a regression model using Keras with TensorFlow as the backend.
###

# ------------- SET DATASET FEATURES HERE -------------- #

FILE_NAME = 'ethylene_methane_ds_10hz.parquet'    # Shape of this dataset is [time_s, methane_ppm, ethylene_ppm, feature1, ..., feature16]

INPUTS = 16                                       # Number of features to be used for training the model
X_COLUMNS = list(range(3, 3+INPUTS))              # Columns to be used as features 
Y_COLUMN = 1                                      # Column to be used as the target variable 
T_COLUMN = 0                                      # Column representing time (in seconds)
FREQUENCY = 10                                    # Sampling frequency of the dataset (in Hz)

# -------- SET DATASET HANDLING OPTIONS HERE ----------- #

WINDOW_SIZE = 200                                 # Size of the sliding window for creating sequences of data for training the model (in seconds)

# ------------------------------------------------------ #

import numpy as np
import os
import tensorflow as tf # Use Keras API from TensorFlow

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR.joinpath('generated-data')

from parquetToPd import parquetToDf

def main(): 
    df = parquetToDf(FILE_NAME)
    data = df.to_numpy()


if __name__=="__main__": # run main fct if this file is executed as a script
    main()