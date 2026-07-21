import numpy as np
from scipy.fft import rfft


def extract_features(segment):
    """
    segment: (window_size, 6)  [acc_x,y,z, gyro_x,y,z]
    return: feature vector
    """
    features = []

    # ========== 时域特征 ==========
    for i in range(segment.shape[1]):
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

    # ========== 轴间相关性 (仅加速度) ==========
    for i in range(3):
        for j in range(i + 1, 3):
            features.append(np.corrcoef(segment[:, i], segment[:, j])[0, 1])

    # ========== 频域特征 (仅加速度) ==========
    for i in range(3):
        fft_vals = np.abs(rfft(segment[:, i]))
        features.extend([
            np.sum(fft_vals),
            np.mean(fft_vals),
            np.max(fft_vals),
            np.argmax(fft_vals),
        ])

    return np.array(features)