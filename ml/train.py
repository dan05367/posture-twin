"""訓練腳本 v2（第 5 週）：
- --data 可以放多個資料夾（錄製的 data/raw + 使用者回饋的 data/feedback）
- --register-if-better：通過把關（gating）才註冊新版本，不會讓變差的模型上線
- --summary-json：把結果寫成 JSON，給自動重訓練服務讀取
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from gating import decide_registration
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score)
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]

# 用「相對於個人基準的偏差」當特徵，這樣不同人 / 不同鏡頭距離都能共用同一個模型
FEATURES = ["dev_shoulder_tilt", "dev_head_roll", "dev_head_offset",
            "dev_head_drop", "dev_shrug", "dev_lean"]
LABELS = ["good", "slouch", "lean_forward", "head_tilt", "shoulder_uneven"]

# 與 edge/posture.py 的 THRESHOLDS 相同（之後可抽成共用套件）
THRESH = {"shoulder_tilt": 5.0, "head_roll": 8.0, "head_offset": 0.15,
          "head_drop": 0.12, "shrug": 0.10, "lean": 1.15}


def load(data_dirs):
    files = sorted(f for d in data_dirs if Path(d).is_dir() for f in Path(d).glob("*.csv"))
    if not files:
        raise SystemExit(f"{list(map(str, data_dirs))} 裡找不到 CSV，請先錄製資料（或跑 make_fake_data.py）")
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


def champion_cv_f1(model_name: str):
    """目前已註冊的最新版本，當時訓練記錄的 best_macro_f1；沒有就回傳 None。"""
    client = MlflowClient()
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            return None
        latest = max(versions, key=lambda v: int(v.version))
        run = client.get_run(latest.run_id)
        parent = run.data.tags.get("mlflow.parentRunId")
        source = client.get_run(parent) if parent else run
        return source.data.metrics.get("best_macro_f1")
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=[str(ROOT / "data" / "raw"), str(ROOT / "data" / "feedback")])
    ap.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    ap.add_argument("--experiment", default="posture-classifier")
    ap.add_argument("--register", action="store_true", help="一律註冊最佳模型（不把關）")
    ap.add_argument("--register-if-better", action="store_true", help="通過把關才註冊")
    ap.add_argument("--min-f1", type=float, default=0.7)
    ap.add_argument("--tolerance", type=float, default=0.02)
    ap.add_argument("--model-name", default="posture-classifier")
    ap.add_argument("--reason", default="manual", help="這次訓練的原因，會記錄成 MLflow tag")
    ap.add_argument("--summary-json", default=None)
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

    with mlflow.start_run(run_name="compare") as parent:
        mlflow.log_params({
            "data_hash": data_hash, "n_files": n_files, "n_rows": len(df),
            "n_sessions": n_sessions, "cv": f"GroupKFold({cv.n_splits})",
            "features": ",".join(FEATURES), "seed": args.seed,
        })
        mlflow.set_tag("trigger_reason", args.reason)
        mlflow.log_dict(df["label"].value_counts().to_dict(), "class_counts.json")

        with mlflow.start_run(run_name="rule_based", nested=True):
            mlflow.log_param("model_type", "rule_based")
            rule_f1 = log_eval(y, rule_predict(df))
        mlflow.log_metric("rule_based_macro_f1", rule_f1)

        scores = {}
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

        champion_f1 = None
        if args.register_if_better:
            champion_f1 = champion_cv_f1(args.model_name)
            ok, why = decide_registration(scores[best], rule_f1, champion_f1, args.min_f1, args.tolerance)
        elif args.register:
            ok, why = True, "--register (no gate)"
        else:
            ok, why = False, "registration not requested"
        mlflow.set_tags({"registered": str(ok), "gate_reason": why})

        version = None
        with mlflow.start_run(run_name=f"final_{best}", nested=True):
            final, _ = models[best]
            final.fit(X, y)
            sig = infer_signature(X, final.predict(X))
            mlflow.log_dict({"features": FEATURES, "labels": LABELS}, "schema.json")
            mlflow.log_param("trained_on_rows", len(df))
            info = mlflow.sklearn.log_model(final, artifact_path="model", signature=sig,
                                            input_example=X.head(3))
            if ok:
                version = mlflow.register_model(info.model_uri, args.model_name).version

        print(f"best model: {best} (macro_f1={scores[best]:.3f})  registered={ok} version={version}  [{why}]")
        if args.summary_json:
            Path(args.summary_json).write_text(json.dumps({
                "registered": ok, "reason": why, "model_version": version,
                "best_model": best, "best_macro_f1": scores[best], "rule_macro_f1": rule_f1,
                "champion_macro_f1": champion_f1, "n_rows": len(df), "n_sessions": n_sessions,
                "data_hash": data_hash, "run_id": parent.info.run_id,
            }, indent=2))


if __name__ == "__main__":
    main()
