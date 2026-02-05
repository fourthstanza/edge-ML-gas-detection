from datagen import generate_dataset_with_features
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR.joinpath('data')

#Modify the variables here 

X, y = generate_dataset_with_features(
    n_samples=4,
    leak_fraction=0.5,
    duration_seconds=180,
    sampling_hz=2.0,
    peak_ppm=200,
    rise_seconds=30,
    fall_seconds=45,
    noise_std=0.06,
    stat_window_seconds=8
)

inp = {""}
yes = {"y","Y"}
no = {"n","N"}
nleaks = sum(y)
while(inp.isdisjoint(yes | no) == True):
    print(f'Dataset generated. Total files to be created: {len(y)}, with {nleaks} leaks and {len(y)-nleaks} baseline. Generate csv files? [y/n]')
    inp = {input()}

if (inp.isdisjoint(yes) == False):
    print(f"Generating CSV files in {DATA_DIR}")
    for i in range(len(y)):
        leak = "leak" if bool(y[i]) else "noleak"
        X[i].to_csv(path_or_buf=DATA_DIR.joinpath(f'{leak}{i}' if bool(y[i]) else f'{leak}{i-nleaks}'), index=False)
