import statistics

from one_euro_filter import OneEuroFilter


def test_reduces_jitter_when_still():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    raw = [10 + (1 if i % 2 else -1) for i in range(200)]
    out = [f(x, i / 30) for i, x in enumerate(raw)]
    assert statistics.pstdev(out[50:]) < 0.3 * statistics.pstdev(raw[50:])


def test_follows_a_real_step_quickly():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.05)
    out = [f(0.0 if i < 30 else 10.0, i / 30) for i in range(90)]
    assert out[-1] > 9.0
