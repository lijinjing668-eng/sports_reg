import numpy as np
from scipy.fft import rfft


def _extract_3axis_features(segment):
    """
    对 3 轴传感器数据做完整的时域+轴间相关性+频域特征提取。
    segment: (window_size, 3)
    返回: 48 维特征向量 (11*3 时域 + 3 相关性 + 4*3 频域)
    """
    features = []

    # ========== 时域特征 (每轴 11 个, 共 33 维) ==========
    for i in range(3):
        col = segment[:, i]
        features.extend([
            np.mean(col),
            np.std(col),
            np.var(col),
            np.max(col),
            np.min(col),
            np.ptp(col),
            np.median(col),
            np.percentile(col, 25),
            np.percentile(col, 75),
            np.mean(np.abs(col)),
            np.mean(col ** 2),
        ])

    # ========== 轴间相关性 (3 维) ==========
    for i in range(3):
        for j in range(i + 1, 3):
            features.append(np.corrcoef(segment[:, i], segment[:, j])[0, 1])

    # ========== 频域特征 (每轴 4 个, 共 12 维) ==========
    for i in range(3):
        fft_vals = np.abs(rfft(segment[:, i]))
        features.extend([
            np.sum(fft_vals),
            np.mean(fft_vals),
            np.max(fft_vals),
            np.argmax(fft_vals),
        ])

    return np.array(features)


def extract_acc_features(segment):
    """
    加速度计特征 (48 维)。
    segment: (window_size, 3) [acc_x, acc_y, acc_z]
    """
    return _extract_3axis_features(segment)


def extract_gyro_features(segment):
    """
    陀螺仪特征 (48 维)。
    segment: (window_size, 3) [gyro_x, gyro_y, gyro_z]
    """
    return _extract_3axis_features(segment)


# ============================================
# 向后兼容：旧版 6 轴合并特征 (81 维)
# 仅在没有使用晚融合的旧代码路径中用到
# ============================================
def extract_features(segment):
    """
    segment: (window_size, 6)  [acc_x,y,z, gyro_x,y,z]
    return: 81 维特征向量 — 兼容旧训练/推理代码
    """
    features = []
    for i in range(segment.shape[1]):
        col = segment[:, i]
        features.extend([
            np.mean(col), np.std(col), np.var(col),
            np.max(col), np.min(col), np.ptp(col),
            np.median(col), np.percentile(col, 25), np.percentile(col, 75),
            np.mean(np.abs(col)), np.mean(col ** 2),
        ])
    for i in range(3):
        for j in range(i + 1, 3):
            features.append(np.corrcoef(segment[:, i], segment[:, j])[0, 1])
    for i in range(3):
        fft_vals = np.abs(rfft(segment[:, i]))
        features.extend([
            np.sum(fft_vals), np.mean(fft_vals),
            np.max(fft_vals), np.argmax(fft_vals),
        ])
    return np.array(features)
