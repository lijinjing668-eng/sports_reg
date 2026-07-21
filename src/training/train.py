"""
训练脚本（方案 A：回退强化 RandomForest）

设计要点：
1. 放弃 CNN（小数据量下欠拟合），使用 RandomForest + 手工特征。
2. 诚实评估：用「留一录制交叉验证 (LORO)」而非随机切分。
   —— 每次留出一个完整录制文件做测试，其余训练，避免同一段录制
      的相邻窗口同时出现在训练集和测试集里造成的准确率虚高。
3. 训练 acc / gyro 两个独立模型（晚融合），分别报告：
   - 窗级 LORO 准确率
   - 段级 LORO 准确率（整段多数投票，与实时"停止后判定"一致）
4. 融合权重按两路的 LORO 可靠度自动分配，写入 fusion_weight.txt。
5. 最终模型在全部数据上重训后保存，供后端加载。
"""
import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from preprocess import apply_filter
from features import extract_acc_features, extract_gyro_features
from config import ACC_CONFIG, GYRO_CONFIG, LABEL_MAP

WINDOW = 100      # 2 秒 @ 50Hz
STEP = 50         # 50% 重叠
ORDER = ["walking", "running", "jumping", "high_knees"]


def make_rf():
    return RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=2,
        max_features="sqrt", class_weight="balanced",
        random_state=42, n_jobs=-1,
    )


def load_recordings(data_dir, pattern, cols, extractor):
    """
    按「录制文件」分组加载。
    返回: list of dict{X:(n_win,48), label:中文, rec_id:str}
    每个文件先低通滤波再滑窗，保证与实时推理一致。
    """
    recs = []
    for en in ORDER:
        folder = data_dir / en
        if not folder.exists():
            continue
        label_cn = LABEL_MAP[en]
        for f in sorted(folder.glob(pattern)):
            df = pd.read_csv(f)
            use = [c for c in cols if c in df.columns]
            arr = df[use].values.astype(float)
            # 去 NaN
            arr = arr[~np.isnan(arr).any(axis=1)]
            if len(arr) < WINDOW:
                print(f"  跳过(过短): {en}/{f.name} ({len(arr)}行)")
                continue
            # 逐通道低通滤波 (与训练/推理一致)
            for c in range(arr.shape[1]):
                arr[:, c] = apply_filter(arr[:, c])
            # 滑窗 + 提特征
            X = []
            for i in range(0, len(arr) - WINDOW + 1, STEP):
                X.append(extractor(arr[i:i + WINDOW]))
            if not X:
                continue
            recs.append({"X": np.array(X), "label": label_cn, "rec_id": f"{en}/{f.name}"})
            print(f"  Loaded {en}/{f.name} -> {len(arr)}行 / {len(X)}窗")
    if not recs:
        raise ValueError(f"未找到匹配 '{pattern}' 的数据于 {data_dir}")
    return recs


def loro_eval(recs, name):
    """
    留一录制交叉验证。返回:
      win_acc  : 窗级准确率
      seg_acc  : 段级准确率(整段多数投票)
      y_true_win, y_pred_win : 用于混淆矩阵
    """
    print(f"\n{'-'*60}\n【{name}】留一录制交叉验证 (LORO)")
    y_true_win, y_pred_win = [], []
    seg_correct, seg_total = 0, 0
    for k in range(len(recs)):
        train_idx = [i for i in range(len(recs)) if i != k]
        X_tr = np.vstack([recs[i]["X"] for i in train_idx])
        y_tr = np.concatenate([[recs[i]["label"]] * len(recs[i]["X"]) for i in train_idx])
        X_te = recs[k]["X"]
        y_te = recs[k]["label"]

        clf = make_rf().fit(X_tr, y_tr)
        preds = clf.predict(X_te)

        y_true_win.extend([y_te] * len(preds))
        y_pred_win.extend(list(preds))

        maj = Counter(preds).most_common(1)[0][0]
        ok = (maj == y_te)
        seg_correct += ok
        seg_total += 1
        flag = "OK " if ok else "ERR"
        print(f"   [{flag}] 留出 {recs[k]['rec_id']:22s} 真实={y_te:5s} -> 段判={maj:5s} "
              f"(窗分布:{dict(Counter(preds))})")

    win_acc = accuracy_score(y_true_win, y_pred_win)
    seg_acc = seg_correct / seg_total
    print(f"   >>> 窗级准确率: {win_acc:.4f} | 段级准确率(多数投票): {seg_acc:.4f} ({seg_correct}/{seg_total})")
    return win_acc, seg_acc, np.array(y_true_win), np.array(y_pred_win)


def plot_confusion(y_true, y_pred, ax, title):
    labels_cn = [LABEL_MAP[e] for e in ORDER]
    cm = confusion_matrix(y_true, y_pred, labels=labels_cn)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels_cn, yticklabels=labels_cn, ax=ax, cbar=False)
    ax.set_xlabel("预测")
    ax.set_ylabel("真实")
    ax.set_title(title)


def main():
    root_dir = Path(__file__).resolve().parents[2]
    data_dir = root_dir / "data" / "processed"
    model_dir = root_dir / "src" / "backend" / "models"
    os.makedirs(model_dir, exist_ok=True)

    print("=" * 60)
    print("方案 A: RandomForest + 留一录制交叉验证 (LORO)")
    print("=" * 60)

    # -------- 1. 加载数据 (按录制分组) --------
    print("\n[加载] 加速度数据:")
    acc_recs = load_recordings(data_dir, "acc_*.csv", ACC_CONFIG["cols"], extract_acc_features)
    print("\n[加载] 陀螺仪数据:")
    gyro_recs = load_recordings(data_dir, "gyro_*.csv", GYRO_CONFIG["cols"], extract_gyro_features)

    # -------- 2. 诚实评估 (LORO) --------
    acc_win, acc_seg, acc_yt, acc_yp = loro_eval(acc_recs, "加速度 RF")
    gyro_win, gyro_seg, gyro_yt, gyro_yp = loro_eval(gyro_recs, "陀螺仪 RF")

    # -------- 3. 融合权重 (按段级可靠度分配) --------
    # 段级准确率更贴近实时"停止后判定整段"的场景，用它加权
    w_acc = acc_seg / (acc_seg + gyro_seg) if (acc_seg + gyro_seg) > 0 else 0.5
    w_acc = round(float(np.clip(w_acc, 0.2, 0.8)), 2)  # 限幅，避免单边独裁
    print(f"\n{'-'*60}")
    print(f"融合权重(按段级可靠度): w_acc={w_acc:.2f}  w_gyro={1-w_acc:.2f}")

    # -------- 4. 全量重训并保存最终模型 --------
    X_acc_all = np.vstack([r["X"] for r in acc_recs])
    y_acc_all = np.concatenate([[r["label"]] * len(r["X"]) for r in acc_recs])
    X_gyro_all = np.vstack([r["X"] for r in gyro_recs])
    y_gyro_all = np.concatenate([[r["label"]] * len(r["X"]) for r in gyro_recs])

    acc_model = make_rf().fit(X_acc_all, y_acc_all)
    gyro_model = make_rf().fit(X_gyro_all, y_gyro_all)

    with open(model_dir / "acc_model.pkl", "wb") as f:
        pickle.dump(acc_model, f)
    with open(model_dir / "gyro_model.pkl", "wb") as f:
        pickle.dump(gyro_model, f)
    with open(model_dir / "fusion_weight.txt", "w") as f:
        f.write(f"{w_acc:.4f}")

    # -------- 5. 混淆矩阵 (LORO 窗级) --------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_confusion(acc_yt, acc_yp, axes[0], f"加速度 RF (LORO窗级 {acc_win:.1%})")
    plot_confusion(gyro_yt, gyro_yp, axes[1], f"陀螺仪 RF (LORO窗级 {gyro_win:.1%})")
    fig.suptitle("留一录制交叉验证混淆矩阵 (诚实泛化)", fontsize=13)
    fig.tight_layout()
    cm_path = data_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=120, bbox_inches="tight")
    print(f"\n混淆矩阵已保存: {cm_path}")

    # -------- 6. 汇总 --------
    print(f"\n{'='*60}")
    print("【最终结果汇总 (LORO 诚实泛化)】")
    print(f"{'指标':<18}{'加速度':>12}{'陀螺仪':>12}")
    print("-" * 42)
    print(f"{'窗级准确率':<18}{acc_win:>11.1%}{gyro_win:>12.1%}")
    print(f"{'段级准确率(投票)':<18}{acc_seg:>11.1%}{gyro_seg:>12.1%}")
    print("-" * 42)
    print(f"融合权重 w_acc={w_acc:.2f}")
    print(f"\n模型已保存到: {model_dir}")
    print("下一步: cd src/backend && python main.py")


if __name__ == "__main__":
    main()
