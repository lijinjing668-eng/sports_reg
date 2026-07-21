"""
晚融合分类器（方案 A：RandomForest + 手工特征）

关键设计：
1. 只用 RF（acc_model.pkl / gyro_model.pkl），不再依赖 TensorFlow/CNN。
2. 修复「训练/推理尺度不匹配」bug：
   推理时把就近数据切成 100 帧窗口，逐窗提取 48 维特征并预测概率，
   再对所有窗口取平均概率 —— 与训练侧 (window=100) 完全一致。
   （旧版把整段当一个大窗提特征，过零率等特征量级差十几倍，导致偏判。）
3. 加置信度阈值：融合置信度低于阈值时返回「不确定」，不硬判。
4. 双流独立累积 (acc_buffer / gyro_buffer)，停止后分别推理再加权融合。
"""
import numpy as np
import pickle
from pathlib import Path
from scipy.signal import butter, filtfilt


# ---- 与训练一致的低通滤波 ----
def _butter_lowpass(cutoff=20, fs=50, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype="low", analog=False)
    return b, a


_B, _A = _butter_lowpass()


def _lowpass(col):
    # filtfilt 需要长度 > 3*max(len(a),len(b))，窗口 100 帧足够
    if len(col) <= 12:
        return col
    return filtfilt(_B, _A, col)


class Classifier:
    def __init__(self):
        self.acc_model = None
        self.gyro_model = None
        self.fusion_weight = 0.5

        # ---- 判定参数 ----
        self.min_duration = 10.0     # 最少 10 秒
        self.recent_frames = 500     # 就近窗口: 最近 10 秒 (500 帧 @50Hz)
        self.window = 100            # 与训练一致
        self.stride = 50             # 与训练一致 (50% 重叠)
        self.conf_threshold = 0.45   # 融合置信度阈值，低于则判"不确定"

        # ---- Buffer ----
        self.acc_buffer = []
        self.gyro_buffer = []
        self.session_start_ts = None
        self.session_end_ts = None

        model_dir = Path(__file__).resolve().parent / "models"
        self._load_models(model_dir)

        weight_path = model_dir / "fusion_weight.txt"
        if weight_path.exists():
            try:
                self.fusion_weight = float(weight_path.read_text().strip())
                print(f"[Classifier] 融合权重 w_acc = {self.fusion_weight:.4f}")
            except Exception:
                pass

    # ---------------------------------------------------------------
    def _load_models(self, model_dir):
        for name, attr in [("acc_model.pkl", "acc_model"), ("gyro_model.pkl", "gyro_model")]:
            path = model_dir / name
            if path.exists():
                try:
                    with open(path, "rb") as f:
                        setattr(self, attr, pickle.load(f))
                    print(f"[Classifier] 模型已加载: {name}")
                except Exception as e:
                    print(f"[Classifier] 加载失败 ({name}): {e}")
            else:
                print(f"[Classifier] 模型缺失: {name}")
        if self.acc_model is None and self.gyro_model is None:
            print("[Classifier] 无可用模型，将使用回退规则判定")

    # ---------------------------------------------------------------
    def reset_session(self):
        self.acc_buffer = []
        self.gyro_buffer = []
        self.session_start_ts = None
        self.session_end_ts = None

    # ---------------------------------------------------------------
    def predict(self, packet):
        """接收手机数据包，分别累积到 acc_buffer / gyro_buffer。"""
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
    # 特征提取 (48 维，与 training/features.py 严格一致)
    # ---------------------------------------------------------------
    @staticmethod
    def _extract_features(window):
        arr = np.asarray(window, dtype=float)
        feats = []
        for i in range(3):
            col = arr[:, i]
            feats.extend([
                np.mean(col), np.std(col), np.var(col),
                np.max(col), np.min(col), np.ptp(col),
                np.median(col),
                np.percentile(col, 25), np.percentile(col, 75),
                np.mean(np.abs(col)), np.mean(col ** 2),
            ])
        for i in range(3):
            for j in range(i + 1, 3):
                c = np.corrcoef(arr[:, i], arr[:, j])[0, 1]
                feats.append(0.0 if np.isnan(c) else c)
        for i in range(3):
            fft_vals = np.abs(np.fft.rfft(arr[:, i]))
            feats.extend([
                np.sum(fft_vals), np.mean(fft_vals),
                np.max(fft_vals),
                float(np.argmax(fft_vals)) if len(fft_vals) > 0 else 0.0,
            ])
        return feats

    def _predict_stream(self, model, buffer_data):
        """
        修复尺度 bug：切成 100 帧窗口逐窗预测，平均概率。
        返回 (pred_label, confidence, proba_vector) 或 (None, 0, None)
        """
        if model is None or len(buffer_data) < self.window:
            return None, 0.0, None

        data = np.array(buffer_data, dtype=float)
        recent = data[-min(len(data), self.recent_frames):]

        # 逐通道低通滤波 (与训练一致)
        filt = recent.copy()
        for c in range(filt.shape[1]):
            filt[:, c] = _lowpass(filt[:, c])

        # 异常值裁剪 (5-sigma, 温和清洗)
        for c in range(filt.shape[1]):
            col = filt[:, c]
            mean_c, std_c = np.mean(col), np.std(col)
            if std_c > 1e-9:
                clip_mask = np.abs(col - mean_c) > 5.0 * std_c
                filt[clip_mask, c] = mean_c

        # 切窗
        windows = []
        for i in range(0, len(filt) - self.window + 1, self.stride):
            windows.append(filt[i:i + self.window])
        if not windows:
            windows = [filt[-self.window:]]

        feats = np.array([self._extract_features(w) for w in windows])
        probas = model.predict_proba(feats)      # (n_win, n_classes)
        avg = probas.mean(axis=0)
        idx = int(np.argmax(avg))
        return model.classes_[idx], float(avg[idx]), avg

    # ---------------------------------------------------------------
    def end_session(self):
        n_acc = len(self.acc_buffer)
        n_gyro = len(self.gyro_buffer)

        duration_sec = None
        if self.session_start_ts and self.session_end_ts:
            duration_sec = (self.session_end_ts - self.session_start_ts) / 1000.0

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
                "message": f"录制时长 {dur_display:.1f}s < {self.min_duration:.0f}s，请延长采集",
                "sample_count": n_acc,
                "duration_sec": round(dur_display, 1),
                "mode": "insufficient",
                "acc_pred": None, "gyro_pred": None,
            }

        result = {
            "event": "session_result",
            "sample_count": n_acc,
            "duration_sec": round(duration_sec, 1) if duration_sec else round(n_acc / 50.0, 1),
            "acc_pred": None, "gyro_pred": None,
            "acc_conf": 0.0, "gyro_conf": 0.0,
        }

        pred_acc, conf_acc, proba_acc = self._predict_stream(self.acc_model, self.acc_buffer)
        pred_gyro, conf_gyro, proba_gyro = self._predict_stream(self.gyro_model, self.gyro_buffer)

        if pred_acc is not None:
            result["acc_pred"] = pred_acc
            result["acc_conf"] = round(conf_acc, 3)
        if pred_gyro is not None:
            result["gyro_pred"] = pred_gyro
            result["gyro_conf"] = round(conf_gyro, 3)

        # ---- 融合 ----
        activity, confidence, mode = None, 0.0, "rule"
        if proba_acc is not None and proba_gyro is not None \
                and np.array_equal(self.acc_model.classes_, self.gyro_model.classes_):
            w = self.fusion_weight
            fused = w * proba_acc + (1 - w) * proba_gyro
            classes = self.acc_model.classes_
            activity = classes[int(np.argmax(fused))]
            confidence = float(np.max(fused))
            mode = "rf_fusion"
        elif proba_acc is not None:
            activity, confidence, mode = pred_acc, conf_acc, "rf_acc_only"
        elif proba_gyro is not None:
            activity, confidence, mode = pred_gyro, conf_gyro, "rf_gyro_only"
        else:
            # 无模型回退规则
            mag = np.linalg.norm(np.mean(np.array(self.acc_buffer[-100:]), axis=0)) \
                if len(self.acc_buffer) > 0 else 0
            if mag > 25:
                activity = "跳跃"
            elif mag > 18:
                activity = "跑步"
            elif mag > 14:
                activity = "高抬腿"
            else:
                activity = "走路"
            confidence, mode = 0.5, "rule"

        # ---- 置信度阈值 ----
        if mode != "rule" and confidence < self.conf_threshold:
            result["activity"] = "不确定"
            result["confidence"] = round(confidence, 3)
            result["mode"] = mode + "_lowconf"
            result["message"] = (f"置信度 {confidence:.0%} 偏低，建议延长采集或重试"
                                 f"（加速度→{result['acc_pred']} / 陀螺仪→{result['gyro_pred']}）")
        else:
            result["activity"] = activity
            result["confidence"] = round(confidence, 3)
            result["mode"] = mode

        return result
