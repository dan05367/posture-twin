"""Week 2 訓練腳本：
讀取錄製的 CSV -> 依 session 分組交叉驗證 -> 比較 規則式 / LogisticRegression / RandomForest
-> 全部記錄到 MLflow -> 用最佳模型在全部資料上重訓並（選擇性）註冊到 Model Registry。


true data
python train.py --data /mnt/d/MLProject/data
"""
import argparse
import hashlib
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score)
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# 用「相對於個人基準的偏差」當特徵，這樣不同人 / 不同鏡頭距離都能共用同一個模型
FEATURES = ["dev_shoulder_tilt", "dev_head_roll", "dev_head_offset",
            "dev_head_drop", "dev_shrug", "dev_lean"]
LABELS = ["good", "slouch", "lean_forward", "head_tilt", "shoulder_uneven"]

# 與 edge/posture.py 的 THRESHOLDS 相同（之後可抽成共用套件）
THRESH = {"shoulder_tilt": 5.0, "head_roll": 8.0, "head_offset": 0.15,
          "head_drop": 0.12, "shrug": 0.10, "lean": 1.15}


def load(data_dir: str):
    files = sorted(Path(data_dir).glob("*.csv"))
    if not files:
        raise SystemExit(f"{data_dir} 裡找不到 CSV，請先錄製資料（或跑 make_fake_data.py）")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    md5 = hashlib.md5()
    for f in files:
        md5.update(f.read_bytes())
    df = df[df["label"].isin(LABELS)].dropna(subset=FEATURES)
    return df, md5.hexdigest()[:12], len(files)


def rule_predict(df: pd.DataFrame) -> np.ndarray:
    """把 Week 1 的規則轉成分類結果，作為要打敗的 baseline。"""
    t = THRESH
    conds = [
        -df["dev_head_drop"] > t["head_drop"],
        df["dev_lean"] > t["lean"],
        (df["dev_head_roll"].abs() > t["head_roll"]) | (df["dev_head_offset"].abs() > t["head_offset"]),
        df["dev_shoulder_tilt"].abs() > t["shoulder_tilt"],
    ]
    return np.select(conds, ["slouch", "lean_forward", "head_tilt", "shoulder_uneven"], default="good")


def log_eval(y, pred) -> float:
    labels = [l for l in LABELS if l in set(y) | set(pred)]
    macro_f1 = f1_score(y, pred, average="macro", labels=labels, zero_division=0)
    mlflow.log_metric("macro_f1", macro_f1)
    mlflow.log_metric("accuracy", accuracy_score(y, pred))
    rep = classification_report(y, pred, labels=labels, output_dict=True, zero_division=0)
    for lab in labels:
        mlflow.log_metric(f"f1_{lab}", rep[lab]["f1-score"])
    mlflow.log_text(classification_report(y, pred, labels=labels, zero_division=0),
                    "classification_report.txt")
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y, pred, labels=labels, normalize="true",
                                            xticks_rotation=30, ax=ax, colorbar=False)
    fig.tight_layout()
    mlflow.log_figure(fig, "confusion_matrix.png")
    plt.close(fig)
    return macro_f1


def build_models(seed: int):
    return {
        "logreg": (
            make_pipeline(StandardScaler(),
                          LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")),
            {"C": 1.0, "class_weight": "balanced"},
        ),
        "random_forest": (
            RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=5,
                                   class_weight="balanced", n_jobs=-1, random_state=seed),
            {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 5,
             "class_weight": "balanced"},
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../../data/raw")
    ap.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    ap.add_argument("--experiment", default="posture-classifier")
    ap.add_argument("--register", action="store_true", help="把最佳模型註冊到 Model Registry")
    ap.add_argument("--model-name", default="posture-classifier")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df, data_hash, n_files = load(args.data)
    X, y, groups = df[FEATURES], df["label"].to_numpy(), df["session"].to_numpy()
    n_sessions = len(set(groups))
    if n_sessions < 2:
        raise SystemExit("至少需要 2 個 session 才能做分組交叉驗證，請多錄幾次（每次重開程式）")
    cv = GroupKFold(n_splits=min(5, n_sessions))

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name="compare"):
        mlflow.log_params({
            "data_hash": data_hash, "n_files": n_files, "n_rows": len(df),
            "n_sessions": n_sessions, "cv": f"GroupKFold({cv.n_splits})",
            "features": ",".join(FEATURES), "seed": args.seed,
        })
        mlflow.log_dict(df["label"].value_counts().to_dict(), "class_counts.json")

        scores = {}
        with mlflow.start_run(run_name="rule_based", nested=True):
            mlflow.log_param("model_type", "rule_based")
            rule_f1 = log_eval(y, rule_predict(df))
        mlflow.log_metric("rule_based_macro_f1", rule_f1)

        models = build_models(args.seed)
        for name, (model, params) in models.items():
            with mlflow.start_run(run_name=name, nested=True):
                mlflow.log_param("model_type", name)
                mlflow.log_params(params)
                pred = cross_val_predict(model, X, y, groups=groups, cv=cv)
                scores[name] = log_eval(y, pred)
                print(f"{name:14s} macro_f1={scores[name]:.3f}")
        print(f"{'rule_based':14s} macro_f1={rule_f1:.3f}")

        best = max(scores, key=scores.get)
        mlflow.log_param("best_model", best)
        mlflow.log_metric("best_macro_f1", scores[best])

        with mlflow.start_run(run_name=f"final_{best}", nested=True):
            final, _ = models[best]
            final.fit(X, y)
            sig = infer_signature(X, final.predict(X))
            mlflow.log_dict({"features": FEATURES, "labels": LABELS}, "schema.json")
            mlflow.log_param("trained_on_rows", len(df))
            mlflow.sklearn.log_model(
                final, artifact_path="model", signature=sig, input_example=X.head(3),
                registered_model_name=args.model_name if args.register else None,
            )
        print(f"best model: {best}  (macro_f1={scores[best]:.3f})")


if __name__ == "__main__":
    main()
