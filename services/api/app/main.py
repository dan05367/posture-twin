import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from .predictor import load_predictor
from .schema import FeaturePayload, Prediction
from .twin import TwinState

logging.basicConfig(level=logging.INFO)

PREDICTIONS = Counter("posture_predictions_total", "Predictions by label", ["label"])
LATENCY = Histogram("posture_inference_seconds", "Model inference latency in seconds")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = load_predictor()
    app.state.twin = TwinState()
    yield


app = FastAPI(title="Posture Twin API", version="0.1.0", lifespan=lifespan)


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
    stable = app.state.twin.update(label, probs, payload.deviations)
    return Prediction(label=label, stable_label=stable, confidence=probs[label],
                      probabilities=probs, model_source=app.state.predictor.source)


@app.get("/twin")
def twin():
    """Digital twin 目前的狀態（之後 Three.js 前端會讀這個）。"""
    return app.state.twin.snapshot()


@app.post("/reload")
def reload_model():
    """重新從 MLflow 載入模型（訓練並註冊新版本後呼叫）。"""
    app.state.predictor = load_predictor()
    return {"model_loaded": app.state.predictor.is_model, "model_source": app.state.predictor.source}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
