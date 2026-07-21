"""
训练脚本（方案 A+：数据清洗 + HGB vs RF 对比选优）

改动：
1. 数据清洗：掐掉首尾 2 秒异动 + 逐通道 4-sigma 异常值裁剪。
2. 模型对比：HistGradientBoostingClassifier (sklearn 内置, 无需额外安装)
   vs RandomForest，LORO 段级准确率高的作为最终模型保存。
3. 写入 model_choice.txt 供后端读取。
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

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from preprocess import apply_filter, trim_and_clean
from features import extract_acc_features, extract_gyro_features
from config import ACC_CONFIG, GYRO_CONFIG, LABEL_MAP

WINDOW = 100
STEP = 50
ORDER = ["walking", "running", "jumping", "high_knees"]

def make_rf():
    return RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=2,
        max_features="sqrt", class_weight="balanced",
        random_state=42, n_jobs=-1,
    )

def make_hgb():
    return HistGradientBoostingClassifier(
        max_iter=300, max_depth=None, min_samples_leaf=5,
        learning_rate=0.08, random_state=42,
        class_weight="balanced",
    )

MODEL_FACTORIES = {"RandomForest": make_rf, "HistGradientBoosting": make_hgb}


def load_recordings(data_dir, pattern, cols, extractor):
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
            arr = arr[~np.isnan(arr).any(axis=1)]
            if len(arr) < WINDOW + 200:  # 留清洗余量
                print(f"  跳过(过短): {en}/{f.name} ({len(arr)}行)")
                continue
            # 清洗：掐头去尾 + 异常值裁剪
            arr = trim_and_clean(arr, trim_sec=2.0, fs=50, sigma=4.0)
            if len(arr) < WINDOW:
                print(f"  跳过(清洗后过短): {en}/{f.name} ({len(arr)}行)")
                continue
            # 逐通道低通滤波
            for c in range(arr.shape[1]):
                arr[:, c] = apply_filter(arr[:, c])
            # 滑窗 + 提特征
            X = []
            for i in range(0, len(arr) - WINDOW + 1, STEP):
                X.append(extractor(arr[i:i + WINDOW]))
            if not X:
                continue
            recs.append({"X": np.array(X), "label": label_cn, "rec_id": f"{en}/{f.name}"})
            print(f"  Loaded {en}/{f.name} -> {len(arr)}行(清洗后) / {len(X)}窗")
    if not recs:
        raise ValueError(f"未找到匹配 '{pattern}' 的数据于 {data_dir}")
    return recs


def loro_eval(recs, model_factory, name):
    y_true_win, y_pred_win = [], []
    seg_correct, seg_total = 0, 0
    for k in range(len(recs)):
        train_idx = [i for i in range(len(recs)) if i != k]
        X_tr = np.vstack([recs[i]["X"] for i in train_idx])
        y_tr = np.concatenate([[recs[i]["label"]] * len(recs[i]["X"]) for i in train_idx])
        X_te = recs[k]["X"]
        y_te = recs[k]["label"]

        clf = model_factory().fit(X_tr, y_tr)
        preds = clf.predict(X_te)

        y_true_win.extend([y_te] * len(preds))
        y_pred_win.extend(list(preds))

        maj = Counter(preds).most_common(1)[0][0]
        ok = (maj == y_te)
        seg_correct += ok
        seg_total += 1
        flag = "OK " if ok else "ERR"
        print(f"   [{flag}] 留出 {recs[k]['rec_id']:22s} 真实={y_te:5s} -> {maj:5s} "
              f"(分布:{dict(Counter(preds))})")

    win_acc = accuracy_score(y_true_win, y_pred_win)
    seg_acc = seg_correct / seg_total
    print(f"   >>> {name} 窗级: {win_acc:.4f} | 段级(投票): {seg_acc:.4f} ({seg_correct}/{seg_total})")
    return win_acc, seg_acc, np.array(y_true_win), np.array(y_pred_win)


def plot_confusion(y_true, y_pred, ax, title):
    labels_cn = [LABEL_MAP[e] for e in ORDER]
    cm = confusion_matrix(y_true, y_pred, labels=labels_cn)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels_cn, yticklabels=labels_cn, ax=ax, cbar=False)
    ax.set_xlabel("预测"); ax.set_ylabel("真实"); ax.set_title(title)


def main():
    root_dir = Path(__file__).resolve().parents[2]
    data_dir = root_dir / "data" / "processed"
    model_dir = root_dir / "src" / "backend" / "models"
    os.makedirs(model_dir, exist_ok=True)

    print("=" * 60)
    print("训练脚本: 数据清洗 + HGB vs RF 对比选优")
    print("=" * 60)

    # ---- 1. 加载 ----
    print("\n[加载] 加速度数据:")
    acc_recs = load_recordings(data_dir, "acc_*.csv", ACC_CONFIG["cols"], extract_acc_features)
    print("\n[加载] 陀螺仪数据:")
    gyro_recs = load_recordings(data_dir, "gyro_*.csv", GYRO_CONFIG["cols"], extract_gyro_features)

    # ---- 2. 多模型对比 ----
    best_acc_name, best_acc_seg, best_gyro_name, best_gyro_seg = None, 0, None, 0
    results = {}

    for model_name, factory in MODEL_FACTORIES.items():
        print(f"\n{'='*60}\n【{model_name}】加速计 LORO:")
        aw, a_seg, a_yt, a_yp = loro_eval(acc_recs, factory, f"加速计 {model_name}")
        print(f"【{model_name}】陀螺仪 LORO:")
        gw, g_seg, g_yt, g_yp = loro_eval(gyro_recs, factory, f"陀螺仪 {model_name}")
        results[model_name] = {
            "acc_win": aw, "acc_seg": a_seg, "acc_yt": a_yt, "acc_yp": a_yp,
            "gyro_win": gw, "gyro_seg": g_seg, "gyro_yt": g_yt, "gyro_yp": g_yp,
        }
        if a_seg > best_acc_seg:
            best_acc_seg, best_acc_name = a_seg, model_name
        if g_seg > best_gyro_seg:
            best_gyro_seg, best_gyro_name = g_seg, model_name

    # 选融合段级最好的
    def fusion_score(name):
        r = results[name]
        return (r["acc_seg"] + r["gyro_seg"]) / 2
    best_fusion = max(MODEL_FACTORIES, key=lambda n: fusion_score(n))
    best_r = results[best_fusion]

    print(f"\n{'='*60}")
    print(f"🏆 最优模型: {best_fusion}")
    print(f"   加速计段级: {best_r['acc_seg']:.1%} | 陀螺仪段级: {best_r['gyro_seg']:.1%}")

    # ---- 3. 融合权重 ----
    w_acc = best_r["acc_seg"] / (best_r["acc_seg"] + best_r["gyro_seg"]) \
        if (best_r["acc_seg"] + best_r["gyro_seg"]) > 0 else 0.5
    w_acc = round(float(np.clip(w_acc, 0.2, 0.8)), 2)
    print(f"   融合权重 w_acc = {w_acc:.2f}")

    # ---- 4. 全量重训并保存 ----
    X_acc_all = np.vstack([r["X"] for r in acc_recs])
    y_acc_all = np.concatenate([[r["label"]] * len(r["X"]) for r in acc_recs])
    X_gyro_all = np.vstack([r["X"] for r in gyro_recs])
    y_gyro_all = np.concatenate([[r["label"]] * len(r["X"]) for r in gyro_recs])

    final_factory = MODEL_FACTORIES[best_fusion]
    acc_model = final_factory().fit(X_acc_all, y_acc_all)
    gyro_model = final_factory().fit(X_gyro_all, y_gyro_all)

    with open(model_dir / "acc_model.pkl", "wb") as f:
        pickle.dump(acc_model, f)
    with open(model_dir / "gyro_model.pkl", "wb") as f:
        pickle.dump(gyro_model, f)
    with open(model_dir / "fusion_weight.txt", "w") as f:
        f.write(f"{w_acc:.4f}")
    with open(model_dir / "model_choice.txt", "w") as f:
        f.write(f"{best_fusion}\n{w_acc:.4f}")

    # ---- 5. 混淆矩阵 ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_confusion(best_r["acc_yt"], best_r["acc_yp"], axes[0],
                   f"加速度 {best_fusion} (LORO窗级 {best_r['acc_win']:.1%})")
    plot_confusion(best_r["gyro_yt"], best_r["gyro_yp"], axes[1],
                   f"陀螺仪 {best_fusion} (LORO窗级 {best_r['gyro_win']:.1%})")
    fig.suptitle("留一录制交叉验证混淆矩阵 (数据已清洗)", fontsize=13)
    fig.tight_layout()
    cm_path = data_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=120, bbox_inches="tight")
    print(f"\n混淆矩阵已保存: {cm_path}")

    # ---- 6. 汇总 ----
    print(f"\n{'='*60}")
    print("【模型对比汇总 (LORO 诚实泛化, 数据已清洗)】")
    print(f"{'模型':<22}{'加速度段级':>12}{'陀螺仪段级':>12}{'综合':>12}")
    print("-" * 60)
    for name, r in results.items():
        total = (r["acc_seg"] + r["gyro_seg"]) / 2
        mark = " ← 选用" if name == best_fusion else ""
        print(f"{name:<22}{r['acc_seg']:>11.1%}{r['gyro_seg']:>12.1%}{total:>11.1%}{mark}")
    print("-" * 60)
    print(f"\n模型已保存: {model_dir}")
    print("下一步: cd src/backend && python main.py")


if __name__ == "__main__":
    main()
