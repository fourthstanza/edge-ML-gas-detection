from pathlib import Path
import numpy as np
import os

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR.joinpath('data')

# Change this variable to pick a different dataset to convert
NAME = 'leak0.csv'
numcols = 5

DATA_TO_CONV = DATA_DIR.joinpath(NAME)

data = np.loadtxt(DATA_TO_CONV, delimiter=',', usecols=(range(0,numcols)), skiprows=1)

f = open(f'{DATA_DIR}/data.h', "x")
f.write(f'// Data file formatted for interpretation by the code within the C++ deployment file.\nfloat data[{len(data)*numcols}] = ' + "{")

last = data[-1][-1]
for row in data:
    for elem in row:
        if elem == last:
            f.write(f'{elem}')
        else:
            f.write(f'{elem}, ')
    f.write('\n')
f.write("};")