import numpy as np
import pickle
import os


class Classifier:
    def __init__(self, model_path="models/rf_model.pkl"):
        self.model = None
        self.session_buffer = []      # 整段录制的所有帧: [acc_x..acc_z, gyro_x..gyro_z]
        self.session_start_ts = None
        self.session_end_ts = None
        self.min_duration = 10.0      # 就近最少 10 秒
        self.max_frames = 1500        # 特征窗口上限 30s @ 50Hz（就近窗口）

        if os.path.exists(model_path):
            try:
                with open(model_path, "rb") as f:
                    self.model = pickle.load(f)
                print(f"[Classifier] Model loaded: {model_path}")
            except Exception as e:
                print(f"[Classifier] Load failed: {e}")

    def reset_session(self):
        """手机端开始发送时调用，清空上一段录制。"""
        self.session_buffer = []
        self.session_start_ts = None
        self.session_end_ts = None

    def _extract_features(self, window):
        arr = np.array(window)  # (N, 6)
        features = []

        # 时域
        for i in range(arr.shape[1]):
            col = arr[:, i]
            features.extend([
                np.mean(col), np.std(col), np.var(col),
                np.max(col), np.min(col), np.ptp(col),
                np.median(col), np.percentile(col, 25), np.percentile(col, 75),
                np.mean(np.abs(col)), np.mean(col ** 2),
            ])

        # 加速度轴间相关性
        for i in range(3):
            for j in range(i + 1, 3):
                features.append(np.corrcoef(arr[:, i], arr[:, j])[0, 1])

        # 频域 (加速度)
        for i in range(3):
            fft_vals = np.abs(np.fft.rfft(arr[:, i]))
            features.extend([
                np.sum(fft_vals), np.mean(fft_vals),
                np.max(fft_vals), np.argmax(fft_vals),
            ])

        return features

    def predict(self, packet):
        """实时数据包：仅累积到 session_buffer，返回原始数据用于前端实时显示。
        不再做实时步态判定（数据太短不可靠）。"""
        acc = packet.get("acc", [0, 0, 0])
        gyro = packet.get("gyro", [0, 0, 0])
        self.session_buffer.append(acc + gyro)
        ts = packet.get("timestamp")
        if ts is not None:
            if self.session_start_ts is None:
                self.session_start_ts = ts
            self.session_end_ts = ts
        return {
            "event": "data",
            "acc": acc,
            "gyro": gyro,
            "gps": packet.get("gps", {}),
            "timestamp": ts,
        }

    def end_session(self):
        """手机端停止发送时调用：用整段录制（就近窗口）做最终步态判定。"""
        n = len(self.session_buffer)

        # 真实时长（手机时钟毫秒）
        duration_sec = None
        if self.session_start_ts and self.session_end_ts:
            duration_sec = (self.session_end_ts - self.session_start_ts) / 1000.0

        # 最少 10 秒判定（优先按时长，否则按帧数兜底）
        too_short = (duration_sec is not None and duration_sec < self.min_duration) or \
                    (duration_sec is None and n < self.min_duration * 50)
        if too_short:
            return {
                "event": "session_result",
                "activity": "数据不足",
                "confidence": 0.0,
                "message": f"录制时长 {(duration_sec if duration_sec is not None else n/50):.1f}s < {self.min_duration:.0f}s，无法判定",
                "sample_count": n,
                "duration_sec": round(duration_sec, 1) if duration_sec is not None else round(n / 50, 1),
            }

        # 就近窗口：超长录制只取最近 max_frames 帧
        window = self.session_buffer[-self.max_frames:]
        features = self._extract_features(window)

        if self.model:
            pred = self.model.predict([features])[0]
            proba = self.model.predict_proba([features])[0]
            return {
                "event": "session_result",
                "activity": pred,
                "confidence": round(float(np.max(proba)), 3),
                "mode": "model",
                "sample_count": n,
                "duration_sec": round(duration_sec, 1) if duration_sec is not None else round(n / 50, 1),
            }

        # 无模型回退规则
        mag = np.linalg.norm(np.mean(np.array(window)[:, :3], axis=0))
        if mag > 25:
            activity = "跳跃"
        elif mag > 18:
            activity = "跑步"
        elif mag > 14:
            activity = "高抬腿"
        else:
            activity = "走路"
        return {
            "event": "session_result",
            "activity": activity,
            "confidence": 0.6,
            "mode": "rule",
            "sample_count": n,
            "duration_sec": round(duration_sec, 1) if duration_sec is not None else round(n / 50, 1),
        }
