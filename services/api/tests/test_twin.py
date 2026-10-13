from app.twin import TwinState


def test_bad_time_accumulates_and_streak_resets():
    t = TwinState(vote=1)
    t.update("good", {}, {}, now=100)
    t.update("slouch", {}, {}, now=101)
    t.update("slouch", {}, {}, now=102)
    snap = t.snapshot(now=102)
    assert snap["bad_seconds_total"] == 2.0
    assert snap["bad_streak_sec"] == 1.0
    t.update("good", {}, {}, now=103)
    snap = t.snapshot(now=103)
    assert snap["bad_streak_sec"] == 0.0
    assert snap["bad_seconds_total"] == 2.0


def test_voting_smooths_single_glitch():
    t = TwinState(vote=5)
    for i in range(4):
        t.update("good", {}, {}, now=i)
    stable = t.update("slouch", {}, {}, now=4)
    assert stable == "good"


def test_gap_is_capped():
    t = TwinState(vote=1, max_gap=5.0)
    t.update("slouch", {}, {}, now=0)
    t.update("slouch", {}, {}, now=100)
    assert t.snapshot(now=100)["bad_seconds_total"] == 5.0


def test_window_drops_old_samples():
    t = TwinState(vote=1, window_sec=10)
    t.update("slouch", {}, {}, now=0)
    t.update("good", {}, {}, now=50)
    assert t.snapshot(now=50)["window_counts"] == {"good": 1}
