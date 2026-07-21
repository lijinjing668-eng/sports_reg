"""
轻量级 1D-CNN 模型：不需要 GPU，CPU 上 2-3 分钟即可训练完成。
原始 (100, 3) 信号直接输入，无需手工特征提取。
"""
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import (
        Conv1D, MaxPooling1D, GlobalAveragePooling1D,
        Dense, Dropout, BatchNormalization, ReLU, Input
    )
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    HAS_TF = True
except ImportError:
    HAS_TF = False


def build_cnn(input_shape=(100, 3), num_classes=4):
    """
    轻量 1D-CNN，约 5 万参数。
    架构: 3 层 Conv1D → GlobalAvgPool → Dense → Softmax
    """
    model = Sequential([
        Input(shape=input_shape, name="input"),
        # Block 1: 捕捉局部模式 (步态周期内的小波动)
        Conv1D(32, 7, padding='same', name="conv1"),
        BatchNormalization(name="bn1"),
        ReLU(name="relu1"),
        MaxPooling1D(2, name="pool1"),
        Dropout(0.2, name="drop1"),

        # Block 2: 中层特征 (跨步幅的模式)
        Conv1D(64, 5, padding='same', name="conv2"),
        BatchNormalization(name="bn2"),
        ReLU(name="relu2"),
        MaxPooling1D(2, name="pool2"),
        Dropout(0.2, name="drop2"),

        # Block 3: 高层特征 (整窗级别的模式)
        Conv1D(128, 3, padding='same', name="conv3"),
        BatchNormalization(name="bn3"),
        ReLU(name="relu3"),
        GlobalAveragePooling1D(name="gap"),

        # 分类头
        Dense(64, activation='relu', name="fc1"),
        Dropout(0.3, name="drop_fc"),
        Dense(num_classes, activation='softmax', name="output"),
    ], name="gait_cnn")

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def train_cnn(segments, labels, model_name, save_dir, epochs=60, batch_size=16):
    """
    训练一个 CNN 模型并保存。

    参数:
        segments: (N, 100, 3) — 滑窗切好的原始信号段
        labels:   (N,) — 中文标签，如 "走路"
        model_name: "acc" 或 "gyro"
        save_dir: 模型保存目录

    返回:
        (model, scaler, encoder, val_acc)
    """
    if not HAS_TF:
        raise ImportError("TensorFlow 未安装，请运行: pip install tensorflow")

    n_samples = len(segments)
    print(f"\n{'='*55}")
    print(f"[CNN-{model_name}] 训练开始")
    print(f"  样本数: {n_samples}, 窗口形状: {segments.shape[1:]}")

    # --- 标准化 ---
    # 将所有窗口展平为 (N*100, 3)，对每通道 (x/y/z) 独立标准化
    segments_flat = segments.reshape(-1, 3)
    scaler = StandardScaler().fit(segments_flat)
    X = scaler.transform(segments_flat).reshape(n_samples, 100, 3)

    # --- 标签编码 ---
    le = LabelEncoder()
    y = le.fit_transform(labels)  # 中文 → 0/1/2/3
    print(f"  类别: {list(le.classes_)}")
    print(f"  各类样本数: {dict(zip(le.classes_, np.bincount(y)))}")

    # --- 构建模型 ---
    model = build_cnn(input_shape=(100, 3), num_classes=len(le.classes_))
    model.summary()

    # --- 回调 ---
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    ]

    # --- 训练 ---
    history = model.fit(
        X, y,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    val_acc = max(history.history['val_accuracy'])
    train_acc = max(history.history['accuracy'])
    print(f"\n[CNN-{model_name}] 训练完成: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")

    # --- 保存 ---
    os.makedirs(save_dir, exist_ok=True)
    model.save(os.path.join(save_dir, f'{model_name}_cnn.keras'), save_format='keras')
    with open(os.path.join(save_dir, f'{model_name}_cnn_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(save_dir, f'{model_name}_cnn_encoder.pkl'), 'wb') as f:
        pickle.dump(le, f)
    print(f"  模型已保存: {save_dir}/{model_name}_cnn.keras")

    return model, scaler, le, val_acc


def predict_with_cnn(model, scaler, encoder, windows, fusion_weight_file=None):
    """
    用 CNN 对多个窗口做预测，返回平均概率和融合权重。

    参数:
        model: Keras 模型
        scaler: StandardScaler
        encoder: LabelEncoder
        windows: (n_windows, 100, 3)

    返回:
        (pred_label, confidence, proba_vector)
    """
    if windows is None or len(windows) == 0:
        return None, 0.0, None

    # 标准化
    n_win, w_len, n_ch = windows.shape
    windows_flat = windows.reshape(-1, n_ch)
    windows_scaled = scaler.transform(windows_flat).reshape(n_win, w_len, n_ch)

    # 预测
    proba = model.predict(windows_scaled, verbose=0)  # (n_win, n_classes)
    avg_proba = proba.mean(axis=0)
    pred_idx = np.argmax(avg_proba)
    confidence = float(np.max(avg_proba))
    pred_label = encoder.inverse_transform([pred_idx])[0]

    return pred_label, confidence, avg_proba
