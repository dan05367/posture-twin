from types import SimpleNamespace as NS

import pytest

from features import FEATURE_NAMES, extract
from posture import evaluate

W = H = 640
BASE = {0: (0.50, 0.30),                      # nose
        2: (0.47, 0.27), 5: (0.53, 0.27),     # eyes
        7: (0.43, 0.28), 8: (0.57, 0.28),     # ears
        11: (0.40, 0.60), 12: (0.60, 0.60)}   # shoulders


def make_lm(overrides=None, scale=1.0, vis=1.0):
    pts = {**BASE, **(overrides or {})}
    lm = [NS(x=0.5, y=0.5, visibility=vis) for _ in range(33)]
    for i, (x, y) in pts.items():
        lm[i] = NS(x=0.5 + scale * (x - 0.5), y=0.5 + scale * (y - 0.5), visibility=vis)
    return lm


def test_symmetric_pose_has_zero_tilt_and_offset():
    f = extract(make_lm(), W, H)
    assert set(f) == set(FEATURE_NAMES)
    assert f["shoulder_tilt"] == pytest.approx(0, abs=1e-6)
    assert f["head_roll"] == pytest.approx(0, abs=1e-6)
    assert f["head_offset"] == pytest.approx(0, abs=1e-6)


def test_lower_right_shoulder_gives_positive_tilt():
    f = extract(make_lm({12: (0.60, 0.65)}), W, H)
    assert f["shoulder_tilt"] == pytest.approx(14.04, abs=0.1)


def test_normalized_features_do_not_depend_on_distance():
    near = extract(make_lm(), W, H)
    far = extract(make_lm(scale=0.5), W, H)
    for k in ("head_offset", "head_drop", "shrug", "shoulder_tilt"):
        assert far[k] == pytest.approx(near[k], abs=1e-6)
    assert far["shoulder_w"] == pytest.approx(near["shoulder_w"] / 2, rel=1e-6)


def test_low_visibility_returns_none():
    assert extract(make_lm(vis=0.1), W, H) is None


def test_evaluate_good_vs_slouch():
    base = extract(make_lm(), W, H)
    _, issues = evaluate(base, base)
    assert issues == []
    slouched = extract(make_lm({0: (0.50, 0.40)}), W, H)  # 鼻子往下
    _, issues = evaluate(slouched, base)
    assert "slouching / head down" in issues
