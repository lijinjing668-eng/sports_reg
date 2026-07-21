# ============================================
# 适配演示数据的配置 (旧版 6 轴合并, 向后兼容)
# ============================================
DEMO_CONFIG = {
    "time_col": "timestamp",
    "acc_cols": ["acc_x", "acc_y", "acc_z"],
    "gyro_cols": ["gyro_x", "gyro_y", "gyro_z"],
    "gps_cols": [],
    "sample_rate": 50,
}

# ============================================
# 晚融合：加速度计独立配置 (processed 数据)
# ============================================
ACC_CONFIG = {
    "time_col": "timestamp",
    "cols": ["acc_x", "acc_y", "acc_z"],
    "sample_rate": 50,
}

# ============================================
# 晚融合：陀螺仪独立配置 (processed 数据)
# ============================================
GYRO_CONFIG = {
    "time_col": "timestamp",
    "cols": ["gyro_x", "gyro_y", "gyro_z"],
    "sample_rate": 50,
}

# 当前激活配置（供旧代码兼容, 新代码直接使用 ACC_CONFIG / GYRO_CONFIG）
ACTIVE_CONFIG = DEMO_CONFIG

# 数据文件夹名 -> 中文标签
LABEL_MAP = {
    "walking": "走路",
    "running": "跑步",
    "jumping": "跳跃",
    "high_knees": "高抬腿",
}
