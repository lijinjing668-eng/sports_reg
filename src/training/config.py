# # ============================================
# # 根据你使用的第三方APP修改此配置
# # ============================================

# # SensorLog (iOS) 默认列名
# SENSORLOG = {
#     "time_col": "loggingTime",
#     "acc_cols": ["accelerometerAccelerationX", "accelerometerAccelerationY", "accelerometerAccelerationZ"],
#     "gyro_cols": ["gyroRotationX", "gyroRotationY", "gyroRotationZ"],
#     "gps_cols": ["locationLatitude", "locationLongitude"],
#     "sample_rate": 50,
# }

# # Physics Toolbox Suite (Android) 列名示例
# PHYSICS_TOOLBOX = {
#     "time_col": "time",
#     "acc_cols": ["Ax", "Ay", "Az"],
#     "gyro_cols": ["Gx", "Gy", "Gz"],
#     "gps_cols": ["Latitude", "Longitude"],
#     "sample_rate": 50,
# }

# # =================== 修改这里 ===================
# ACTIVE_CONFIG = SENSORLOG  # 或 PHYSICS_TOOLBOX

# # 数据文件夹名 -> 中文标签
# LABEL_MAP = {
#     "walking": "走路",
#     "running": "跑步",
#     "jumping": "跳跃",
#     "high_knees": "高抬腿",
# }


# ============================================
# 适配演示数据的配置
# ============================================

DEMO_CONFIG = {
    "time_col": "timestamp",
    "acc_cols": ["acc_x", "acc_y", "acc_z"],
    "gyro_cols": ["gyro_x", "gyro_y", "gyro_z"],
    "gps_cols": [],  # 演示数据无GPS
    "sample_rate": 50,
}

ACTIVE_CONFIG = DEMO_CONFIG

# 数据文件夹名 -> 中文标签
LABEL_MAP = {
    "walking": "走路",
    "running": "跑步",
    "jumping": "跳跃",
    "high_knees": "高抬腿",
}