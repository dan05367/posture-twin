import csv
import tempfile
from datetime import datetime
from pathlib import Path

from app.feedback import COLUMNS, save_feedback
from app.schema import DEV_NAMES


def rows(n, label="slouch"):
    return [{"ts": 1000.0 + i, "stable_label": label, **{k: 0.1 * (i + 1) for k in DEV_NAMES}} for i in range(n)]


def test_creates_file_with_header_and_appends():
    with tempfile.TemporaryDirectory() as d:
        when = datetime(2026, 10, 14, 12, 0, 0)
        n, path = save_feedback(Path(d) / "fb", "good", rows(2), when)
        assert n == 2 and path.name == "feedback_20261014.csv"
        save_feedback(Path(d) / "fb", "head_tilt", rows(1), when)
        data = list(csv.DictReader(open(path, encoding="utf-8")))
        assert list(data[0].keys()) == COLUMNS
        assert [r["label"] for r in data] == ["good", "good", "head_tilt"]
        assert data[0]["session"] == "feedback_20261014" and data[0]["predicted"] == "slouch"
