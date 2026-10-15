from gating import decide_registration


def test_first_model_passes_when_good_enough():
    ok, _ = decide_registration(0.85, 0.70, None)
    assert ok


def test_rejects_below_minimum():
    ok, why = decide_registration(0.60, 0.40, None, min_f1=0.7)
    assert not ok and "minimum" in why


def test_rejects_when_not_better_than_rules():
    ok, why = decide_registration(0.80, 0.82, None)
    assert not ok and "rule-based" in why


def test_rejects_clear_regression_vs_champion():
    ok, why = decide_registration(0.80, 0.60, 0.90, tolerance=0.02)
    assert not ok and "worse" in why


def test_accepts_small_dip_within_tolerance_and_improvements():
    assert decide_registration(0.89, 0.60, 0.90, tolerance=0.02)[0]
    assert decide_registration(0.95, 0.60, 0.90)[0]
