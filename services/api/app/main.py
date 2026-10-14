import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from .predictor import load_predictor
from .schema import DEV_NAMES, LABELS, FeaturePayload, Prediction
from .twin import TwinState

logging.basicConfig(level=logging.INFO)

# ---- Prometheus 指標 ----
PREDICTIONS = Counter("posture_predictions_total", "Predictions by label", ["label"])
LATENCY = Histogram("posture_inference_seconds", "Model inference latency in seconds")
CONFIDENCE = Histogram("posture_confidence", "Max class probability of each prediction",
                       buckets=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0))
DEVIATION = Gauge("posture_deviation", "Latest deviation from calibrated baseline", ["feature"])
STABLE = Gauge("posture_stable_label", "1 for the current stable label, else 0", ["label"])


def _twin_value(key: str, default: float = float("nan")) -> float:
    """scrape 當下才去讀 twin 狀態；沒有資料時回傳 default。"""
    twin = getattr(app.state, "twin", None)
    if twin is None:
        return default
    snap = twin.snapshot()
    return snap[key] if snap.get("has_data") else default


Gauge("posture_seconds_since_last_sample", "Seconds since the last sample from Edge") \
    .set_function(lambda: _twin_value("age_sec"))
Gauge("posture_window_bad_ratio", "Share of non-good predictions in the last 10 minutes") \
    .set_function(lambda: _twin_value("window_bad_ratio"))
Gauge("posture_bad_streak_seconds", "Seconds of continuous bad posture") \
    .set_function(lambda: _twin_value("bad_streak_sec", 0.0))
Gauge("posture_bad_seconds", "Cumulative bad-posture seconds since API start") \
    .set_function(lambda: _twin_value("bad_seconds_total", 0.0))
Gauge("posture_model_loaded", "1 if an MLflow model is loaded, 0 if using rule fallback") \
    .set_function(lambda: 1.0 if getattr(getattr(app.state, "predictor", None), "is_model", False) else 0.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = load_predictor()
    app.state.twin = TwinState()
    yield


app = FastAPI(title="Posture Twin API", version="0.2.0", lifespan=lifespan)


@app.get("/health")
def health():
    p = app.state.predictor
    return {"status": "ok", "model_loaded": p.is_model, "model_source": p.source}


@app.post("/features", response_model=Prediction)
def post_features(payload: FeaturePayload):
    t0 = time.perf_counter()
    label, probs = app.state.predictor.predict(payload.deviations)
    LATENCY.observe(time.perf_counter() - t0)
    PREDICTIONS.labels(label=label).inc()
    CONFIDENCE.observe(probs[label])

    stable = app.state.twin.update(label, probs, payload.deviations)

    for k in DEV_NAMES:
        DEVIATION.labels(feature=k).set(payload.deviations[k])
    for l in LABELS:
        STABLE.labels(label=l).set(1.0 if l == stable else 0.0)

    return Prediction(label=label, stable_label=stable, confidence=probs[label],
                      probabilities=probs, model_source=app.state.predictor.source)


@app.get("/twin")
def twin():
    """Digital twin 目前的狀態（之後 Three.js 前端會讀這個）。"""
    return app.state.twin.snapshot()


@app.get("/recent")
def recent(limit: int = Query(1000, ge=1, le=10000), stable_label: str | None = None):
    """最近的特徵列（只有數值，沒有影像），給 drift 監控服務使用。"""
    rows = app.state.twin.recent(limit, stable_label)
    return {"count": len(rows), "rows": rows}


@app.post("/reload")
def reload_model():
    """重新從 MLflow 載入模型（訓練並註冊新版本後呼叫）。"""
    app.state.predictor = load_predictor()
    return {"model_loaded": app.state.predictor.is_model, "model_source": app.state.predictor.source}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
