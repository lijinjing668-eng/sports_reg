"""
将 Physics Toolbox Sensor Suite 采集的真实数据转换为与模拟数据一致的格式。

原始数据（data/raw 下，陀螺仪与加速度计为两个独立文件）：
    <motion>.csv       : time, wx, wy, wz            (陀螺仪)
    <motion>_a.csv     : time, ax, ay, az, atotal    (加速度计, atotal 为模长冗余列)

转换后（data/processed/<label>/<label>_converted.csv）：
    timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z

处理方式：
    1. 两个文件时间轴不同步、采样率不同（gyro~30-59Hz, acc~100Hz）。
    2. 取两者时间重叠区间，线性插值重采样到统一 50Hz 网格。
    3. 重命名列，丢弃冗余的 atotal。
这样生成的文件可直接被 train.py 的 load_csv_files 原样读取（列顺序与 DEMO_CONFIG 一致）。
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"
TARGET_FS = 50  # Hz, 与 train.py 的 window_size=100 ("2s @ 50Hz") 保持一致

# 原始文件名词干 -> 输出标签文件夹名 (需与 config.LABEL_MAP 对应)
MOTION_MAP = {
    "walking": "walking",
    "running": "running",
    "jumping": "jumping",
    "highknees": "high_knees",
}


def convert_one(gyro_path, acc_path, out_path):
    g = pd.read_csv(gyro_path).sort_values("time").rename(
        columns={"wx": "gyro_x", "wy": "gyro_y", "wz": "gyro_z"}
    )
    a = pd.read_csv(acc_path).sort_values("time").rename(
        columns={"ax": "acc_x", "ay": "acc_y", "az": "acc_z"}
    )
    g = g[["time", "gyro_x", "gyro_y", "gyro_z"]]
    a = a[["time", "acc_x", "acc_y", "acc_z"]]

    # 时间重叠区间，避免边缘 NaN
    t_start = max(a["time"].min(), g["time"].min())
    t_end = min(a["time"].max(), g["time"].max())
    if t_end <= t_start:
        raise ValueError(f"陀螺仪与加速度计无时间重叠: {gyro_path.name} / {acc_path.name}")

    grid = np.arange(t_start, t_end, 1.0 / TARGET_FS)

    out = pd.DataFrame({"timestamp": grid})
    for col in ["acc_x", "acc_y", "acc_z"]:
        out[col] = np.interp(grid, a["time"].values, a[col].values)
    for col in ["gyro_x", "gyro_y", "gyro_z"]:
        out[col] = np.interp(grid, g["time"].values, g[col].values)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return len(out), t_end - t_start


def main():
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"未找到原始数据目录: {RAW_DIR}")

    print(f"输入目录: {RAW_DIR}")
    print(f"输出目录: {OUT_DIR}  (目标采样率 {TARGET_FS}Hz)\n")

    for stem, label in MOTION_MAP.items():
        gyro = RAW_DIR / f"{stem}.csv"
        acc = RAW_DIR / f"{stem}_a.csv"
        if not (gyro.exists() and acc.exists()):
            print(f"[跳过] {label}: 缺少文件 ({gyro.name} / {acc.name})")
            continue
        out_path = OUT_DIR / label / f"{label}_converted.csv"
        n_rows, dur = convert_one(gyro, acc, out_path)
        print(f"[完成] {label:10s} -> {out_path.name:22s} 行数={n_rows:5d}  时长≈{dur:.1f}s")

    print("\n转换完成。下一步: 重跑 train.py 生成新模型 (读取 data/processed)。")


if __name__ == "__main__":
    main()
