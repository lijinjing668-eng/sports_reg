"""
晚融合分类器：维护两个独立的传感器 buffer，停止时分别预测后加权融合。
"""
import numpy as np
import pickle
import os
from pathlib import Path


class Classifier:
    def __init__(self):
        self.acc_model = None
        self.gyro_model = None
        self.fusion_weight = 0.5  # 默认 0.5，train.py 训练后从文件读取

        self.acc_buffer = []
        self.gyro_buffer = []
        self.session_start_ts = None
        self.session_end_ts = None
        self.min_duration = 10.0      # 最少 10 秒
        self.max_frames = 1500        # 就近窗口上限 30s @ 50Hz

        model_dir = Path(__file__).resolve().parent / "models"

        # 加载加速度模型
        acc_path = model_dir / "acc_model.pkl"
        if acc_path.exists():
            try:
                with open(acc_path, "rb") as f:
                    self.acc_model = pickle.load(f)
                print(f"[Classifier] 加速度模型已加载 (classes={list(self.acc_model.classes_)})")
            except Exception as e:
                print(f"[Classifier] 加速度模型加载失败: {e}")
        else:
            print(f"[Classifier] 未找到加速度模型: {acc_path}")

        # 加载陀螺仪模型
        gyro_path = model_dir / "gyro_model.pkl"
        if gyro_path.exists():
            try:
                with open(gyro_path, "rb") as f:
                    self.gyro_model = pickle.load(f)
                print(f"[Classifier] 陀螺仪模型已加载 (classes={list(self.gyro_model.classes_)})")
            except Exception as e:
                print(f"[Classifier] 陀螺仪模型加载失败: {e}")
        else:
            print(f"[Classifier] 未找到陀螺仪模型: {gyro_path}")

        # 读取融合权重
        weight_path = model_dir / "fusion_weight.txt"
        if weight_path.exists():
            try:
                with open(weight_path, "r") as f:
                    self.fusion_weight = float(f.read().strip())
                print(f"[Classifier] 融合权重 w_acc = {self.fusion_weight:.4f}")
            except Exception:
                pass

        if self.acc_model is None and self.gyro_model is None:
            print("[Classifier] 无可用模型，将使用回退规则判定")

    def reset_session(self):
        """手机端开始发送时调用"""
        self.acc_buffer = []
        self.gyro_buffer = []
        self.session_start_ts = None
        self.session_end_ts = None

    # ---------------------------------------------------------------
    # 特征提取（与 features.py 完全一致，避免跨文件导入）
    # ---------------------------------------------------------------
    @staticmethod
    def _extract_3axis_features(window):
        """
        window: (N, 3) — 对任意 3 轴数据提 48 维特征。
        和训练时的 features._extract_3axis_features 保持严格一致。
        """
        arr = np.array(window)
        features = []

        # 时域 (33 维)
        for i in range(3):
            col = arr[:, i]
            features.extend([
                np.mean(col), np.std(col), np.var(col),
                np.max(col), np.min(col), np.ptp(col),
                np.median(col),
                np.percentile(col, 25), np.percentile(col, 75),
                np.mean(np.abs(col)), np.mean(col ** 2),
            ])

        # 轴间相关性 (3 维)
        for i in range(3):
            for j in range(i + 1, 3):
                c = np.corrcoef(arr[:, i], arr[:, j])[0, 1]
                features.append(0.0 if np.isnan(c) else c)

        # 频域 (12 维)
        for i in range(3):
            fft_vals = np.abs(np.fft.rfft(arr[:, i]))
            features.extend([
                np.sum(fft_vals), np.mean(fft_vals),
                np.max(fft_vals),
                np.argmax(fft_vals) if len(fft_vals) > 0 else 0,
            ])

        return features

    # ---------------------------------------------------------------
    # 实时数据累积
    # ---------------------------------------------------------------
    def predict(self, packet):
        """
        接收手机发来的数据包，分别累积到 acc_buffer 和 gyro_buffer。
        不再做实时判定，仅返回原始数据供前端绘图。
        """
        acc = packet.get("acc", [0.0, 0.0, 0.0])
        gyro = packet.get("gyro", [0.0, 0.0, 0.0])
        self.acc_buffer.append([float(v) for v in acc])
        self.gyro_buffer.append([float(v) for v in gyro])

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

    # ---------------------------------------------------------------
    # 停止后判定
    # ---------------------------------------------------------------
    def end_session(self):
        """
        手机停止发送后调用。
        两个模型各自对各自 buffer 做预测，加权融合后返回最终步态。
        """
        n_acc = len(self.acc_buffer)
        n_gyro = len(self.gyro_buffer)

        # 真实时长（手机时钟毫秒）
        duration_sec = None
        if self.session_start_ts and self.session_end_ts:
            duration_sec = (self.session_end_ts - self.session_start_ts) / 1000.0

        # 最少 10 秒
        too_short = (
            (duration_sec is not None and duration_sec < self.min_duration)
            or (duration_sec is None and min(n_acc, n_gyro) < self.min_duration * 50)
        )
        if too_short:
            dur_display = duration_sec if duration_sec else min(n_acc, n_gyro) / 50.0
            return {
                "event": "session_result",
                "activity": "数据不足",
                "confidence": 0.0,
                "message": f"录制时长 {dur_display:.1f}s < {self.min_duration:.0f}s，无法判定",
                "sample_count": n_acc,
                "duration_sec": round(dur_display, 1),
                "acc_pred": None, "gyro_pred": None,
            }

        # 就近窗口
        acc_win = np.array(self.acc_buffer[-self.max_frames:])
        gyro_win = np.array(self.gyro_buffer[-self.max_frames:])

        result = {
            "event": "session_result",
            "sample_count": n_acc,
            "duration_sec": round(duration_sec, 1) if duration_sec else round(n_acc / 50.0, 1),
            "acc_pred": None, "gyro_pred": None,
            "acc_conf": 0.0, "gyro_conf": 0.0,
        }

        # 加速度模型预测
        proba_acc = None
        if self.acc_model is not None and len(acc_win) >= 2:
            try:
                feat_acc = self._extract_3axis_features(acc_win)
                proba_acc = self.acc_model.predict_proba([feat_acc])[0]
                pred_acc = self.acc_model.classes_[np.argmax(proba_acc)]
                result["acc_pred"] = pred_acc
                result["acc_conf"] = round(float(np.max(proba_acc)), 3)
            except Exception as e:
                print(f"[Classifier] 加速度预测失败: {e}")

        # 陀螺仪模型预测
        proba_gyro = None
        if self.gyro_model is not None and len(gyro_win) >= 2:
            try:
                feat_gyro = self._extract_3axis_features(gyro_win)
                proba_gyro = self.gyro_model.predict_proba([feat_gyro])[0]
                pred_gyro = self.gyro_model.classes_[np.argmax(proba_gyro)]
                result["gyro_pred"] = pred_gyro
                result["gyro_conf"] = round(float(np.max(proba_gyro)), 3)
            except Exception as e:
                print(f"[Classifier] 陀螺仪预测失败: {e}")

        # ---- 融合 ----
        if proba_acc is not None and proba_gyro is not None:
            # 两部模型都成功 → 加权融合
            w = self.fusion_weight
            fused = w * proba_acc + (1 - w) * proba_gyro
            result["activity"] = self.acc_model.classes_[np.argmax(fused)]
            result["confidence"] = round(float(np.max(fused)), 3)
            result["mode"] = "fusion"
        elif proba_acc is not None:
            result["activity"] = result["acc_pred"]
            result["confidence"] = result["acc_conf"]
            result["mode"] = "acc_only"
        elif proba_gyro is not None:
            result["activity"] = result["gyro_pred"]
            result["confidence"] = result["gyro_conf"]
            result["mode"] = "gyro_only"
        else:
            # 无模型回退规则
            mag = np.linalg.norm(np.mean(acc_win, axis=0)) if len(acc_win) > 0 else 0
            if mag > 25:
                result["activity"] = "跳跃"
            elif mag > 18:
                result["activity"] = "跑步"
            elif mag > 14:
                result["activity"] = "高抬腿"
            else:
                result["activity"] = "走路"
            result["confidence"] = 0.6
            result["mode"] = "rule"

        return result
