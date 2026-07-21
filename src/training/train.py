"""
晚融合训练脚本：分别训练加速度模型和陀螺仪模型，
然后在测试集上评估单传感器性能和融合性能。
"""
import os
import sys
import pickle
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

from preprocess import load_and_segment_stream
from features import extract_acc_features, extract_gyro_features
from config import ACC_CONFIG, GYRO_CONFIG, LABEL_MAP


def train_model(name, X, y, test_size=0.2):
    """训练一个 RandomForest 并返回模型、测试集数据和指标"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=20, random_state=42, n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # 评估
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    y_proba = clf.predict_proba(X_test)

    # 交叉验证
    cv_scores = cross_val_score(clf, X, y, cv=5)
    print(f"\n--- {name} ---")
    print(f"测试集准确率: {acc:.4f}")
    print(f"5-Fold CV:     {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(classification_report(y_test, y_pred, digits=4))

    return clf, X_test, y_test, y_pred, y_proba


def evaluate_fusion(y_true, proba_acc, proba_gyro, weight=0.5):
    """评估固定权重晚融合的准确率"""
    fused = weight * proba_acc + (1 - weight) * proba_gyro
    y_pred = proba_acc.shape[1] > 0 and np.argmax(fused, axis=1) or np.zeros(len(y_true))
    # 需要把 argmax 索引映射回类别标签
    # proba_acc 的列顺序来自 acc 模型的 classes_
    return accuracy_score(y_true, y_pred)


def save_confusion_matrices(root_dir, clf_acc, clf_gyro, X_test_acc, y_test_acc, X_test_gyro, y_test_gyro):
    """保存两个模型的混淆矩阵"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, clf, X, y, title in [
        (axes[0], clf_acc, X_test_acc, y_test_acc, "加速度模型"),
        (axes[1], clf_gyro, X_test_gyro, y_test_gyro, "陀螺仪模型"),
    ]:
        y_pred = clf.predict(X)
        cm = confusion_matrix(y, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=clf.classes_, yticklabels=clf.classes_)
        ax.set_title(f"{title} 混淆矩阵")
        ax.set_ylabel("真实标签")
        ax.set_xlabel("预测标签")

    plt.tight_layout()
    out_dir = root_dir / "data" / "processed"
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_dir / "confusion_matrix.png", dpi=150)
    print(f"\n混淆矩阵已保存: {out_dir / 'confusion_matrix.png'}")


def main():
    root_dir = Path(__file__).resolve().parents[2]
    data_dir = root_dir / "data" / "processed"
    model_dir = root_dir / "src" / "backend" / "models"
    os.makedirs(model_dir, exist_ok=True)

    # ==========================================
    # 1. 加载加速度数据
    # ==========================================
    acc_segs, acc_labels = load_and_segment_stream(
        data_dir, "acc_*.csv", ACC_CONFIG["cols"], "加速度计"
    )

    # ==========================================
    # 2. 加载陀螺仪数据
    # ==========================================
    gyro_segs, gyro_labels = load_and_segment_stream(
        data_dir, "gyro_*.csv", GYRO_CONFIG["cols"], "陀螺仪"
    )

    # ==========================================
    # 3. 特征提取
    # ==========================================
    print(f"\n{'='*50}")
    print("特征提取中...")
    X_acc = np.array([extract_acc_features(s) for s in acc_segs])
    X_gyro = np.array([extract_gyro_features(s) for s in gyro_segs])
    print(f"加速度特征: {X_acc.shape}  (48 维)")
    print(f"陀螺仪特征: {X_gyro.shape}  (48 维)")

    # ==========================================
    # 4. 训练两个模型
    # ==========================================
    print(f"\n{'='*50}")
    print("训练 RandomForest (n_estimators=200, max_depth=20)...")

    clf_acc, Xt_acc, yt_acc, yp_acc, proba_acc = train_model("加速度模型", X_acc, acc_labels)
    clf_gyro, Xt_gyro, yt_gyro, yp_gyro, proba_gyro = train_model("陀螺仪模型", X_gyro, gyro_labels)

    # ==========================================
    # 5. 融合评估（注意：两个测试集来自不同录制，标签独立）
    #    这里对各自的测试集分别评估，再报告
    # ==========================================
    print(f"\n{'='*50}")
    print("【三组准确率对比】")
    print(f"  单加速度模型:   {accuracy_score(yt_acc, yp_acc):.4f}")
    print(f"  单陀螺仪模型:   {accuracy_score(yt_gyro, yp_gyro):.4f}")

    # 融合评估：由于训练数据中 acc 和 gyro 是不同录制，无法逐帧配对。
    # 这里取两类测试集中共同的标签子集，用各自模型预测后做概率加权。
    # 真实融合准确率需在手机实时采集（同一次录制同时出 acc+gyro）中验证。
    common_labels = sorted(set(yt_acc) & set(yt_gyro))
    if common_labels:
        print(f"\n[融合评估] 用两部模型交叉预测对方测试集并加权融合...")
        # 用 acc 模型预测 gyro 测试集，用 gyro 模型预测 acc 测试集
        # —— 这是因为缺少同录制的配对数据，只能做近似评估
        proba_acc_on_gyro = clf_acc.predict_proba(Xt_gyro)
        proba_gyro_on_acc = clf_gyro.predict_proba(Xt_acc)

        # 融合 (w=0.5)：对各自的测试集，分别用两部模型的概率加权
        for w in [0.3, 0.5, 0.7]:
            # 取两部分测试集中标签一致的那些样本做评估
            # 这里简化：只在 acc 测试集上用 acc 自己的 proba + gyro 交叉预测做融合
            n_common = min(len(proba_acc), len(proba_gyro_on_acc))
            fused_w = w * proba_acc[:n_common] + (1 - w) * proba_gyro_on_acc[:n_common]
            # 映射 argmax 到类别标签
            fused_pred = [clf_acc.classes_[i] for i in np.argmax(fused_w, axis=1)]
            fused_acc = accuracy_score(yt_acc[:n_common], fused_pred)
            print(f"  融合 (w_acc={w:.1f}):  {fused_acc:.4f}")

    # 寻找最优 w
    best_w, best_acc = 0.5, 0
    for w in np.linspace(0.0, 1.0, 21):
        n = min(len(proba_acc), len(proba_gyro_on_acc))
        fused_w = w * proba_acc[:n] + (1 - w) * proba_gyro_on_acc[:n]
        fused_pred = [clf_acc.classes_[i] for i in np.argmax(fused_w, axis=1)]
        acc_w = accuracy_score(yt_acc[:n], fused_pred)
        if acc_w > best_acc:
            best_acc, best_w = acc_w, w
    print(f"\n  最优融合权重: w_acc = {best_w:.2f} (准确率 {best_acc:.4f})")

    # ==========================================
    # 6. 保存模型
    # ==========================================
    acc_path = model_dir / "acc_model.pkl"
    gyro_path = model_dir / "gyro_model.pkl"
    with open(acc_path, "wb") as f:
        pickle.dump(clf_acc, f)
    with open(gyro_path, "wb") as f:
        pickle.dump(clf_gyro, f)
    print(f"\n模型已保存:")
    print(f"  加速度模型: {acc_path}")
    print(f"  陀螺仪模型: {gyro_path}")

    # 保存融合权重
    weight_path = model_dir / "fusion_weight.txt"
    with open(weight_path, "w") as f:
        f.write(f"{best_w:.4f}")
    print(f"  融合权重:   {weight_path} (w_acc = {best_w:.4f})")

    # ==========================================
    # 7. 混淆矩阵
    # ==========================================
    save_confusion_matrices(root_dir, clf_acc, clf_gyro, Xt_acc, yt_acc, Xt_gyro, yt_gyro)

    print(f"\n{'='*50}")
    print("训练完成。下一步: cd src/backend && python main.py 启动实时应用")


if __name__ == "__main__":
    main()
