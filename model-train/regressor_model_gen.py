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

# ------------------------------------------------------ #

import numpy as np
import random
import tensorflow as tf # Use Keras API from TensorFlow

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR.joinpath('generated-data')

from parquet_to_pd import parquetToDf
from csv_to_pd import csvToDf

class DataSplit:

    features: np.ndarray
    target: np.ndarray
    time: np.ndarray | None

    data_length: int

    tr_ind: list[int]
    tst_ind: list[int]
    val_ind: list[int]

    winlen: int

    def __init__(self, X: np.ndarray, y:np.ndarray, win_length: int, split: tuple[float, float, float] | None, time: np.ndarray | None = None, seed: int | None = None):

        if win_length > X.shape[0]:
            raise ValueError("Length of window must be smaller than length of dataset")

        self.winlen = win_length
        self.features = X
        self.target = y
        
        self.time = time

        self.data_length = self.features.shape[0]

        if split is not None:
            if abs(sum(split) - 1.0) > 1e-6:
                raise ValueError("Split proportions must sum to 1")
            if split[0] <= 0 or split[1] <= 0 or split[2] <= 0:
                raise ValueError("Split proportions must be positive")
            if split[1] * self.data_length-win_length < 1 or split[2] * self.data_length-win_length < 1:
                raise ValueError("Split proportions too small, no data allocated to test or validation set")
        else:
            split  = (0.6, 0.2, 0.2)

        self.split(split, seed)

    def split(self, split: tuple[float, float, float], seed: int | None = None):
        
        if seed is None:
            seed = random.randint(0, 2 ** 32)

        random.seed(seed)

        # data starts at end of first window
        datarange = list(range(self.winlen,self.data_length))

        # choose k random indices w/o replacement for training
        self.tr_ind = random.sample(datarange, k = int(len(datarange) * split[0]))
        # remaining indices are for testing & validation
        remaining = list(set(datarange) - set(self.tr_ind))

        # proportion of remaining data to be used for testing vs validation
        prop = split[1] / (split[1] + split[2]) 
        self.tst_ind = random.sample(remaining, k = int(len(remaining) * prop))
        self.val_ind = list(set(remaining) - set(self.tst_ind))

    def tr_length(self) -> int:
        return len(self.tr_ind)
    
    def tst_length(self) -> int:
        return len(self.tst_ind)

    def val_length(self) -> int:
        return len(self.val_ind)

    def Xtr_i(self, i: int) -> np.ndarray | float:
        ind = self.tr_ind[i]
        if len(self.features.shape) == 1:
            return self.features[ind-self.winlen:ind]
        else:
            return self.features[ind-self.winlen:ind, :]

    def ytr_i(self, i: int) -> np.ndarray | float:
        ind = self.tr_ind[i]
        if len(self.target.shape) == 1:
            return self.target[ind]
        else:
            return self.target[ind, :]
    
    def Xtst_i(self, i: int) -> np.ndarray | float:
        ind = self.tst_ind[i]
        if len(self.features.shape) == 1:
            return self.features[ind-self.winlen:ind]
        else:
            return self.features[ind-self.winlen:ind, :]

    def ytst_i(self, i: int) -> np.ndarray | float:
        ind = self.tst_ind[i]
        if len(self.target.shape) == 1:
            return self.target[ind]
        else:
            return self.target[ind, :]

    def Xval_i(self, i: int) -> np.ndarray | float:
        ind = self.val_ind[i]
        if len(self.features.shape) == 1:
            return self.features[ind-self.winlen:ind]
        else:
            return self.features[ind-self.winlen:ind, :]
        
    def yval_i(self, i: int) -> np.ndarray | float:
        ind = self.val_ind[i]
        if len(self.target.shape) == 1:
            return self.target[ind]
        else:
            return self.target[ind, :]

    def ttr_i(self, i: int) -> float | None:
        if self.time is None:
            raise ValueError("No time data provided to DataSplit, cannot get time value")
        ind = self.tr_ind[i]
        return self.time[ind]
    
    def ttst_i(self, i: int) -> float | None:
        if self.time is None:
            raise ValueError("No time data provided to DataSplit, cannot get time value")
        ind = self.tst_ind[i]
        return self.time[ind]
    
    def tval_i(self, i: int) -> float | None:
        if self.time is None:
            raise ValueError("No time data provided to DataSplit, cannot get time value")
        ind = self.val_ind[i]
        return self.time[ind]

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

    data_split = create_data_split(data, seed = 42)

    


if __name__=="__main__": # run main fct if this file is executed as a script
    main()