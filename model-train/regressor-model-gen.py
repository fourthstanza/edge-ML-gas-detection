###
# This script is used to generate a regression model for predicting the concentration of a gas (e.g., methane) based on time-series data. 
# It reads a dataset in Apache Parquet format, processes it, and trains a regression model using Keras with TensorFlow as the backend.
# Ensure that the data is complete and has no NaN or Null values, as no handling is done for these cases and the output will be undefined.
###

# ------------- SET DATASET FEATURES HERE -------------- #

FILE_NAME: str = 'ethylene_methane_ds_10hz.parquet'    # Shape of this dataset is [time_s, methane_ppm, ethylene_ppm, feature1, ..., feature16]
PARQUET: bool = True

INPUTS: int = 16                                       # Number of features to be used for training the model
X_COLUMNS: list = list(range(3, 3+INPUTS))              # Column(s) to be used as features 
Y_COLUMNS: list = [1]                                      # Column to be used as the target variable 
T_COLUMN:  int = 0                                      # Column representing time (in seconds)
FREQUENCY: int = 10                                    # Sampling frequency of the dataset (in Hz)

# -------- SET DATASET HANDLING OPTIONS HERE ----------- #

WINDOW_SIZE: int = 200                                 # Size of the sliding window for creating sequences of data for training the model (in seconds)

# ------------------------------------------------------ #

import numpy as np
import random
import os
import tensorflow as tf # Use Keras API from TensorFlow

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR.joinpath('generated-data')

from parquetToPd import parquetToDf

class DataSplit:

    data: np.ndarray
    target: np.ndarray

    data_length: int

    tr_ind: list[int]
    tst_ind: list[int]

    winlen: int

    indices: range

    def __init__(self, X: np.ndarray, y:np.ndarray, win_length: int, split:  float | tuple[float, float], seed: int | None = None):

        if win_length > X.shape[1]:
            raise ValueError("Length of window must be smaller than length of dataset")

        self.winlen = win_length
        self.data = X
        self.target = y
        self.data_length = self.data.shape[1]

        if seed is None:
            seed = np.random.randint(0, 2 ** 32)

        self.indices = range(0, self.data_length-self.winlen)

        if isinstance(split, tuple):
            split = split[0]

        self.split(split, seed)


    def split(self, split: float, seed: int):
        
        np.random.seed(seed)

        len = self.data.shape[0]

        # data starts at end of first window
        datarange = list(range(self.winlen,len))

        self.tr_ind = random.choices(datarange, k = int(self.data_length * split))
        self.tst_ind = list(set(datarange) - set(self.tr_ind))



        
    
    
    


def dataSplit(data: np.ndarray, split: float | tuple[float, float], seed: int | None = None) -> DataSplit:

    winlength = WINDOW_SIZE * FREQUENCY

    X = data[X_COLUMNS]
    y = data[Y_COLUMNS]

    splt = DataSplit(X, y, winlength, split, seed)

    return splt

def main(): 
    df = parquetToDf(FILE_NAME)
    data = df.to_numpy()

    




if __name__=="__main__": # run main fct if this file is executed as a script
    main()