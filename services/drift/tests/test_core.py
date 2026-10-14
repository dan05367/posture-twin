import numpy as np
import pandas as pd

from app.core import DEV_COLS, compute_drift, psi


def frame(rng, n=1000, shifts=None):
    shifts = shifts or {}
    return pd.DataFrame({c: rng.normal(shifts.get(c, 0.0), 1.0, n) for c in DEV_COLS})


def test_psi_is_small_for_same_distribution():
    rng = np.random.default_rng(0)
    assert psi(rng.normal(0, 1, 3000), rng.normal(0, 1, 3000)) < 0.05


def test_psi_is_large_for_shifted_distribution():
    rng = np.random.default_rng(0)
    assert psi(rng.normal(0, 1, 3000), rng.normal(1.5, 1, 3000)) > 0.5


def test_psi_handles_constant_reference_and_nan():
    assert psi([1.0] * 100, [1.0] * 50) == 0.0
    assert np.isnan(psi([], [1.0, 2.0]))
    assert psi([1.0, 2.0, np.nan, 3.0] * 20, [1.0, 2.0, 3.0] * 20) >= 0


def test_no_drift_when_same_distribution():
    rng = np.random.default_rng(1)
    res = compute_drift(frame(rng, 2000), frame(rng, 500))
    assert res["n_drifted"] == 0
    assert res["dataset_drift"] is False


def test_two_shifted_features_are_detected_but_not_dataset_drift():
    rng = np.random.default_rng(2)
    cur = frame(rng, 500, {"dev_shoulder_tilt": 1.5, "dev_head_roll": 1.5})
    res = compute_drift(frame(rng, 2000), cur)
    assert res["n_drifted"] == 2
    assert res["features"]["dev_shoulder_tilt"]["drifted"] is True
    assert res["features"]["dev_head_drop"]["drifted"] is False
    assert res["dataset_drift"] is False          # 2/6 < 0.5


def test_four_shifted_features_flag_dataset_drift():
    rng = np.random.default_rng(3)
    shifts = {c: 1.5 for c in DEV_COLS[:4]}
    res = compute_drift(frame(rng, 2000), frame(rng, 500, shifts))
    assert res["n_drifted"] == 4
    assert res["dataset_drift"] is True           # 4/6 >= 0.5
