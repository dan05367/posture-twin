import pytest
from fastapi.testclient import TestClient

from app.main import app

GOOD = dict(shoulder_tilt=0.5, head_roll=1.0, head_offset=0.01, head_drop=0.0, shrug=0.0, lean=1.0)
SLOUCH = {**GOOD, "head_drop": -0.25}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("MODEL_URI", raising=False)  # 測試一律用規則式備援
    with TestClient(app) as c:
        yield c


def post(client, dev):
    return client.post("/features", json={"deviations": dev})


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is False
    assert body["model_source"] == "rule-based-fallback"


def test_good_posture(client):
    r = post(client, GOOD)
    assert r.status_code == 200
    assert r.json()["label"] == "good"


def test_slouch_posture(client):
    body = post(client, SLOUCH).json()
    assert body["label"] == "slouch"
    assert body["stable_label"] == "slouch"
    assert body["confidence"] == 1.0


def test_accepts_full_edge_payload(client):
    payload = {"ts": 1.0, "features": {"shoulder_w": 0.3}, "deviations": GOOD, "issues": []}
    assert client.post("/features", json=payload).status_code == 200


def test_missing_key_is_422(client):
    bad = {k: v for k, v in GOOD.items() if k != "lean"}
    assert post(client, bad).status_code == 422


def test_twin_before_and_after(client):
    assert client.get("/twin").json() == {"has_data": False}
    for _ in range(3):
        post(client, SLOUCH)
    twin = client.get("/twin").json()
    assert twin["has_data"] is True
    assert twin["window_counts"] == {"slouch": 3}
    assert twin["stable_label"] == "slouch"


def test_metrics_exposed(client):
    post(client, GOOD)
    text = client.get("/metrics").text
    assert "posture_predictions_total" in text
    assert "posture_inference_seconds" in text


def test_reload_falls_back_without_uri(client):
    r = client.post("/reload")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is False
