import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import os
import glob
from config import ACTIVE_CONFIG, LABEL_MAP


def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def apply_filter(data, cutoff=20, fs=50, order=4):
    b, a = butter_lowpass(cutoff, fs, order)
    return filtfilt(b, a, data)


def load_csv_files(data_dir):
    all_data = []
    cfg = ACTIVE_CONFIG

    for label_en, label_cn in LABEL_MAP.items():
        folder = os.path.join(data_dir, label_en)
        if not os.path.exists(folder):
            print(f"Warning: folder not found: {folder}")
            continue
        files = glob.glob(os.path.join(folder, "*.csv"))
        for f in files:
            df = pd.read_csv(f)
            df["label"] = label_cn
            all_data.append(df)
            print(f"  Loaded {f} -> {len(df)} rows")

    if not all_data:
        raise ValueError("No CSV files found! Check your data/raw/ folder structure.")
    return pd.concat(all_data, ignore_index=True)


def preprocess(df):
    cfg = ACTIVE_CONFIG
    cols = cfg["acc_cols"] + cfg["gyro_cols"]

    # 去除缺失值
    df = df.dropna(subset=[c for c in cols if c in df.columns])

    # 低通滤波去噪
    for col in cols:
        if col in df.columns:
            df[col] = apply_filter(df[col].values)

    return df


def segment(df, window_size=100, step_size=50):
    cfg = ACTIVE_CONFIG
    cols = cfg["acc_cols"] + cfg["gyro_cols"]

    segments, labels = [], []
    for i in range(0, len(df) - window_size, step_size):
        window = df.iloc[i:i + window_size]
        segments.append(window[cols].values)
        labels.append(window["label"].iloc[0])

    return np.array(segments), np.array(labels)