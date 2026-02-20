# Code to generate dummy data for training an edge impulse model
# Modify parameters & run from main.py

# X shape -> (n_samples, n_timesteps, n_channels)
#
# Channels:
# 0 -> methane_ppm (raw)
# 1 -> rolling_mean
# 2 -> rolling_rms
# 3 -> rolling_std
# 4 -> rolling_slope

import numpy as np
import pandas as pd

def rolling_mean(x, w):
    return np.convolve(x, np.ones(w) / w, mode="same")

def rolling_rms(x, w):
    return np.sqrt(rolling_mean(x**2, w))

def rolling_std(x, w):
    mu = rolling_mean(x, w)
    return np.sqrt(rolling_mean((x - mu)**2, w))

def rolling_slope(x, w):
    slope = np.zeros_like(x)
    half = w // 2

    for i in range(len(x)):
        start = max(0, i - half)
        end = min(len(x), i + half + 1)
        t = np.arange(end - start)
        y = x[start:end]

        if len(y) > 1:
            slope[i] = np.polyfit(t, y, 1)[0]
    return slope

rng = np.random.default_rng(42)

def generate_methane_timeseries(
    duration_seconds=120,
    sampling_hz=1.0,
    baseline_ppm=2.0,
    peak_ppm=50.0,
    rise_seconds=20,
    fall_seconds=30,
    noise_std=0.05,
    leak=True
):
    n = int(duration_seconds * sampling_hz)
    methane = np.full(n, baseline_ppm)

    if leak:
        max_start = n - int((rise_seconds + fall_seconds) * sampling_hz) - 1
        leak_start = rng.integers(int(0.1 * n), max_start)

        rise_n = int(rise_seconds * sampling_hz)
        fall_n = int(fall_seconds * sampling_hz)

        for i in range(rise_n):
            idx = leak_start + i
            methane[idx] = baseline_ppm + (
                peak_ppm - baseline_ppm
            ) * (1 - np.exp(-3 * i / rise_n))

        for i in range(fall_n):
            idx = leak_start + rise_n + i
            methane[idx] = baseline_ppm + (
                peak_ppm - baseline_ppm
            ) * np.exp(-3 * i / fall_n)

    # noise
    noise = np.zeros(n)
    eps = rng.normal(0, noise_std, n)
    phi = 0.85
    for i in range(1, n):
        noise[i] = phi * noise[i - 1] + eps[i]

    methane += noise
    return methane, int(leak)

def generate_dataset_with_features(
    n_samples=10,
    leak_fraction=0.5,
    stat_window_seconds=10,
    **kwargs
):
    X = []
    y = []

    sampling_hz = kwargs.get("sampling_hz", 1.0)
    w = max(3, int(stat_window_seconds * sampling_hz))

    n_leak = int(n_samples * leak_fraction)
    n_no_leak = n_samples - n_leak

    for leak in ([1] * n_leak + [0] * n_no_leak):
        methane, label = generate_methane_timeseries(leak=bool(leak), **kwargs)

        mean = rolling_mean(methane, w)
        rms = rolling_rms(methane, w)
        std = rolling_std(methane, w)
        slope = rolling_slope(methane, w)

        features = pd.DataFrame(
            {
                "ppm" : methane,
                "mean" : mean,
                "rms" : rms,
                "std" : std,
                "slope" : slope,
                "label" : bool(leak)
            }
        )

        X.append(features)
        y.append(label)

    return X, y
