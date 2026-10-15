import csv

import pytest
from fastapi.testclient import TestClient

from app.main import app

GOOD = dict(shoulder_tilt=0.5, head_roll=1.0, head_offset=0.01, head_drop=0.0, shrug=0.0, lean=1.0)
SLOUCH = {**GOOD, "head_drop": -0.25}


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.delenv("MODEL_URI", raising=False)
    monkeypatch.setenv("FEEDBACK_DIR", str(tmp_path))
    with TestClient(app) as c:
        yield c


def post(client, dev, n=1):
    for _ in range(n):
        assert client.post("/features", json={"deviations": dev}).status_code == 200


def test_feedback_writes_labeled_rows(client, tmp_path):
    post(client, SLOUCH, 3)
    r = client.post("/feedback", json={"label": "good", "seconds": 30})
    assert r.status_code == 200 and r.json()["saved"] == 3
    files = list(tmp_path.glob("feedback_*.csv"))
    assert len(files) == 1
    rows = list(csv.DictReader(open(files[0], encoding="utf-8")))
    assert len(rows) == 3
    assert rows[0]["label"] == "good" and "dev_head_drop" in rows[0] and "predicted" in rows[0]


def test_double_click_does_not_duplicate(client):
    post(client, SLOUCH, 2)
    assert client.post("/feedback", json={"label": "slouch"}).status_code == 200
    assert client.post("/feedback", json={"label": "slouch"}).status_code == 409


def test_second_feedback_appends_new_rows_only(client, tmp_path):
    post(client, GOOD, 2)
    client.post("/feedback", json={"label": "good"})
    post(client, SLOUCH, 2)
    assert client.post("/feedback", json={"label": "slouch"}).json()["saved"] == 2
    rows = list(csv.DictReader(open(next(tmp_path.glob("feedback_*.csv")), encoding="utf-8")))
    assert [r["label"] for r in rows] == ["good", "good", "slouch", "slouch"]


def test_no_samples_is_409(client):
    assert client.post("/feedback", json={"label": "good"}).status_code == 409


def test_invalid_label_is_422(client):
    post(client, GOOD)
    assert client.post("/feedback", json={"label": "banana"}).status_code == 422


def test_ui_page_is_served(client):
    r = client.get("/ui")
    assert r.status_code == 200 and "<svg" in r.text
