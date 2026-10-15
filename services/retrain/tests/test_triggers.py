import tempfile
from pathlib import Path

from app.triggers import (StateStore, TriggerConfig, count_rows, evaluate_triggers,
                          should_notify_drift)

CFG = TriggerConfig(min_new_rows=100, min_interval_sec=3600)


def test_count_rows_ignores_header_and_missing_dirs():
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "a.csv").write_text("label,x\ngood,1\ngood,2\n")
        (Path(d) / "b.csv").write_text("label,x\nslouch,3\n")
        (Path(d) / "note.txt").write_text("ignored\n")
        assert count_rows([d, "/definitely/not/here"]) == 3


def test_not_enough_new_rows():
    go, why = evaluate_triggers(150, 100, None, 1000.0, CFG)
    assert not go and "50/100" in why


def test_enough_new_rows_triggers():
    go, why = evaluate_triggers(250, 100, None, 1000.0, CFG)
    assert go and "150" in why


def test_cooldown_blocks_even_with_new_rows():
    go, why = evaluate_triggers(500, 100, last_run_ts=900.0, now=1000.0, cfg=CFG)
    assert not go and "cooling down" in why
    assert evaluate_triggers(500, 100, last_run_ts=900.0, now=900.0 + 3601, cfg=CFG)[0]


def test_force_ignores_everything():
    assert evaluate_triggers(0, 0, 999.0, 1000.0, CFG, force=True) == (True, "manual request")


def test_drift_notification_cooldown():
    assert not should_notify_drift(False, None, 100.0, 3600)
    assert should_notify_drift(True, None, 100.0, 3600)
    assert not should_notify_drift(True, 100.0, 200.0, 3600)
    assert should_notify_drift(True, 100.0, 4000.0, 3600)


def test_state_store_roundtrip_and_missing_file():
    with tempfile.TemporaryDirectory() as d:
        store = StateStore(Path(d) / "sub" / "state.json")
        assert store.load() == {}
        store.save({"trained_rows": 42})
        assert store.load() == {"trained_rows": 42}
