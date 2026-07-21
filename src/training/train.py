import os
import sys
import pickle
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.rcParams["font.sans-serif"] = ["SimHei"]  # 或者 ["Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

from preprocess import load_csv_files, preprocess, segment
from features import extract_features


def main():
    root_dir = Path(__file__).resolve().parents[2]
    data_dir = root_dir / "data" / "processed"  # 真实数据已转换至此 (raw 保留原始采集)
    print("=" * 50)
    print("Step 1: Loading CSV files...")
    df = load_csv_files(data_dir)
    print(f"Total rows: {len(df)}")

    print("\nStep 2: Preprocessing (filtering)...")
    df = preprocess(df)

    print("\nStep 3: Segmenting into windows...")
    segments, labels = segment(df, window_size=100, step_size=50)
    print(f"Total segments: {len(segments)}")

    print("\nStep 4: Extracting features...")
    X = np.array([extract_features(s) for s in segments])
    y = labels
    print(f"Feature dimension: {X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nStep 5: Training Random Forest...")
    clf = RandomForestClassifier(
        n_estimators=200, max_depth=20, random_state=42, n_jobs=-1
    )
    clf.fit(X_train, y_train)

    print("\nStep 6: Evaluating...")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, digits=4))

    scores = cross_val_score(clf, X, y, cv=5)
    print(f"5-Fold CV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

    # 保存模型
    os.makedirs(root_dir / "src" / "backend" / "models", exist_ok=True)
    model_path = root_dir / "src" / "backend" / "models" / "rf_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"\nModel saved to: {model_path}")

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=clf.classes_, yticklabels=clf.classes_)
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    os.makedirs(root_dir / "data" / "processed", exist_ok=True)
    plt.savefig(root_dir / "data" / "processed" / "confusion_matrix.png", dpi=150)
    print("Confusion matrix saved to: ../data/processed/confusion_matrix.png")


if __name__ == "__main__":
    main()