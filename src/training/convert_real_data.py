"""
将 Physics Toolbox 双流数据独立重采样到 50Hz，不再做 acc+gyro 合并。
加速度 (wa*/ra*/ja*/ha*) 和陀螺仪 (wi*/ri*/ji*/hi*) 分别输出到独立 CSV。

原始文件 (data/raw/<motion>/):
    waN.csv : time, ax, ay, az, aT       — 线性加速度
    wiN.csv : time, wx, wy, wz            — 陀螺仪

输出 (data/processed/<motion>/):
    acc_N.csv  : timestamp, acc_x, acc_y, acc_z    (50Hz 重采样)
    gyro_N.csv : timestamp, gyro_x, gyro_y, gyro_z (50Hz 重采样)
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"
TARGET_FS = 50  # Hz, 与 train.py 滑窗 100 帧 = 2s 保持一致

# 文件名 2 字符前缀 → (输出文件夹名, 传感器类型)
# a = accelerometer, i = inertial (gyro)
FILE_MAP = {
    "wa": ("walking",    "acc"),
    "wi": ("walking",    "gyro"),
    "ra": ("running",    "acc"),
    "ri": ("running",    "gyro"),
    "ja": ("jumping",    "acc"),
    "ji": ("jumping",    "gyro"),
    "ha": ("highknees",  "acc"),
    "hi": ("highknees",  "gyro"),
}

# 输出文件夹名 → LABEL_MAP key
OUT_LABELS = {
    "walking":    "walking",
    "running":    "running",
    "jumping":    "jumping",
    "highknees":  "high_knees",
}


def resample_50hz(df, time_col="time", value_cols=None):
    """对单个传感器数据做线性插值重采样到 50Hz"""
    df = df.sort_values(time_col).dropna(subset=[time_col])
    t = df[time_col].values
    if len(t) < 2:
        return None, 0
    t_start, t_end = t[0], t[-1]
    if t_end - t_start < 1.0 / TARGET_FS:
        return None, 0
    grid = np.arange(t_start, t_end, 1.0 / TARGET_FS)
    out = pd.DataFrame({"timestamp": grid})
    for col in value_cols:
        out[col] = np.interp(grid, t, df[col].values)
    return out, t_end - t_start


def main():
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"未找到原始数据目录: {RAW_DIR}")

    print(f"输入: {RAW_DIR}")
    print(f"输出: {OUT_DIR}  (目标采样率 {TARGET_FS}Hz)")
    print("=" * 65)

    total_files = 0
    for folder_name in ["walking", "running", "jumping", "highknees"]:
        raw_folder = RAW_DIR / folder_name
        if not raw_folder.exists():
            print(f"[SKIP] 文件夹不存在: {folder_name}")
            continue

        csv_files = sorted(raw_folder.glob("*.csv"))
        for f in csv_files:
            stem = f.stem.lower()  # e.g. "wa1", "ri3"
            prefix = stem[:2]

            if prefix not in FILE_MAP:
                print(f"[SKIP] {f.name:15s} 未知前缀 '{prefix}'")
                continue

            raw_label, sensor_type = FILE_MAP[prefix]
            if raw_label != folder_name:
                print(f"[WARN] {f.name:15s} 文件夹={folder_name} 但前缀映射到={raw_label}，跳过")
                continue

            out_label = OUT_LABELS[raw_label]

            # 读取并重采样
            try:
                df = pd.read_csv(f)
            except Exception as e:
                print(f"[ERR] {f.name:15s} 读取失败: {e}")
                continue

            if sensor_type == "acc":
                # 列: time, ax, ay, az, aT
                out_df, dur = resample_50hz(df, "time", ["ax", "ay", "az"])
                if out_df is None:
                    print(f"[SKIP] {f.name:15s} 数据太短，无法重采样")
                    continue
                out_df = out_df.rename(columns={"ax": "acc_x", "ay": "acc_y", "az": "acc_z"})
                out_name = f"acc_{stem[2:]}.csv"
            else:
                # 列: time, wx, wy, wz
                out_df, dur = resample_50hz(df, "time", ["wx", "wy", "wz"])
                if out_df is None:
                    print(f"[SKIP] {f.name:15s} 数据太短，无法重采样")
                    continue
                out_df = out_df.rename(columns={"wx": "gyro_x", "wy": "gyro_y", "wz": "gyro_z"})
                out_name = f"gyro_{stem[2:]}.csv"

            out_path = OUT_DIR / out_label / out_name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_df.to_csv(out_path, index=False)
            print(f"[OK]  {sensor_type:4s}  {out_label:12s}  {out_name:12s}  "
                  f"行数={len(out_df):5d}  时长≈{dur:.1f}s")
            total_files += 1

    print("=" * 65)
    print(f"转换完成: 共处理 {total_files} 个文件")
    print(f"输出目录: {OUT_DIR}")
    print(f"下一步: cd src/training && python train.py")


if __name__ == "__main__":
    main()
