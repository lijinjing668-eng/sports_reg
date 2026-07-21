import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import os
import glob
from pathlib import Path
from config import ACTIVE_CONFIG, LABEL_MAP, ACC_CONFIG, GYRO_CONFIG


def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def apply_filter(data, cutoff=20, fs=50, order=4):
    b, a = butter_lowpass(cutoff, fs, order)
    return filtfilt(b, a, data)


# ============================================
# 旧版 6 轴合并加载 (向后兼容, train_old.py 用)
# ============================================
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
        raise ValueError("No CSV files found!")
    return pd.concat(all_data, ignore_index=True)


def preprocess(df):
    cfg = ACTIVE_CONFIG
    cols = cfg["acc_cols"] + cfg["gyro_cols"]
    df = df.dropna(subset=[c for c in cols if c in df.columns])
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


# ============================================
# 晚融合：双流独立加载+预处理+滑窗
# ============================================
def _load_single_stream(data_dir, file_pattern, col_names):
    """
    加载指定模式的 CSV 文件，统一赋予 label。
    data_dir: Path, e.g. Path("data/processed")
    file_pattern: str, e.g. "acc_*.csv" or "gyro_*.csv"
    col_names: list, e.g. ["acc_x","acc_y","acc_z"]
    """
    data_dir = Path(data_dir)
    all_data = []
    for label_en, label_cn in LABEL_MAP.items():
        folder = data_dir / label_en
        if not folder.exists():
            continue
        files = sorted(folder.glob(file_pattern))
        if not files:
            print(f"  [{label_en}] 无匹配文件 ({file_pattern})")
            continue
        for f in files:
            df = pd.read_csv(f)
            # 只保留需要的列
            keep = ["timestamp"] + [c for c in col_names if c in df.columns]
            df = df[keep].copy()
            df["label"] = label_cn
            all_data.append(df)
            print(f"  Loaded {f} -> {len(df)} rows")

    if not all_data:
        raise ValueError(f"No files found matching '{file_pattern}' in {data_dir}")
    return pd.concat(all_data, ignore_index=True)


def preprocess_single_stream(df, col_names):
    """对单流数据做低通滤波去噪"""
    for col in col_names:
        if col in df.columns:
            df = df.dropna(subset=[col])
            df[col] = apply_filter(df[col].values)
    return df


def segment_single_stream(df, col_names, window_size=100, step_size=50):
    """对单流数据滑窗切段, 返回 (segments, labels)"""
    segments, labels = [], []
    for i in range(0, len(df) - window_size, step_size):
        window = df.iloc[i:i + window_size]
        seg = window[[c for c in col_names if c in df.columns]].values
        segments.append(seg)
        labels.append(window["label"].iloc[0])
    return np.array(segments), np.array(labels)


def load_and_segment_stream(data_dir, file_pattern, col_names, stream_name="sensor"):
    """
    一站式：加载 → 预处理 → 滑窗切段。
    返回 (segments, labels)
    """
    print(f"\n{'='*50}")
    print(f"[{stream_name}] 加载数据 (pattern: {file_pattern})")
    df = _load_single_stream(data_dir, file_pattern, col_names)
    print(f"[{stream_name}] 总行数: {len(df)}")
    print(f"[{stream_name}] 预处理（低通滤波 20Hz）...")
    df = preprocess_single_stream(df, col_names)
    print(f"[{stream_name}] 滑窗切段 (window=100, step=50 @50Hz)...")
    segs, labs = segment_single_stream(df, col_names)
    print(f"[{stream_name}] 总段数: {len(segs)}, 标签分布:")
    for lb in sorted(set(labs)):
        print(f"    {lb}: {(labs == lb).sum()}")
    return segs, labs
