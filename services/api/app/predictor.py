import logging
import os

import pandas as pd

from .schema import DEV_NAMES, LABELS

log = logging.getLogger("posture.predictor")
COLUMNS = [f"dev_{k}" for k in DEV_NAMES]

# 與 edge/posture.py、services/training/train.py 相同的門檻
THRESH = {"shoulder_tilt": 5.0, "head_roll": 8.0, "head_offset": 0.15,
          "head_drop": 0.12, "lean": 1.15}


class RulePredictor:
    """沒有載入模型時的備援：Week 1 的規則式判斷。"""
    source = "rule-based-fallback"
    is_model = False

    def predict(self, dev: dict):
        t = THRESH
        if -dev["head_drop"] > t["head_drop"]:
            label = "slouch"
        elif dev["lean"] > t["lean"]:
            label = "lean_forward"
        elif abs(dev["head_roll"]) > t["head_roll"] or abs(dev["head_offset"]) > t["head_offset"]:
            label = "head_tilt"
        elif abs(dev["shoulder_tilt"]) > t["shoulder_tilt"]:
            label = "shoulder_uneven"
        else:
            label = "good"
        return label, {c: 1.0 if c == label else 0.0 for c in LABELS}


class ModelPredictor:
    """包住一個有 predict_proba / classes_ 的 sklearn 模型。"""
    is_model = True

    def __init__(self, model, source: str):
        self.model = model
        self.source = source

    def predict(self, dev: dict):
        X = pd.DataFrame([[float(dev[k]) for k in DEV_NAMES]], columns=COLUMNS)
        proba = self.model.predict_proba(X)[0]
        classes = [str(c) for c in self.model.classes_]
        probs = {c: float(p) for c, p in zip(classes, proba)}
        return max(probs, key=probs.get), probs


def load_predictor():
    """依環境變數 MODEL_URI 從 MLflow 載入模型；失敗就退回規則式，服務不會因此掛掉。"""
    uri = os.getenv("MODEL_URI")
    if not uri:
        log.warning("MODEL_URI not set, using rule-based fallback")
        return RulePredictor()
    try:
        import mlflow.sklearn
        model = mlflow.sklearn.load_model(uri)
        log.info("loaded model from %s", uri)
        return ModelPredictor(model, f"mlflow:{uri}")
    except Exception:
        log.exception("failed to load model from %s, using rule-based fallback", uri)
        return RulePredictor()
