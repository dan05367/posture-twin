from notifier import Notifier


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_cooldown_limits_notifications():
    clock, sent = Clock(), []
    n = Notifier([lambda t, m: sent.append((t, m))], cooldown=300, clock=clock)
    assert n.maybe_notify("a", "b") is True
    clock.t = 100
    assert n.maybe_notify("a", "b") is False
    clock.t = 301
    assert n.maybe_notify("a", "b") is True
    assert len(sent) == 2


def test_failing_backend_does_not_break_others():
    sent = []

    def broken(title, message):
        raise RuntimeError("no display")

    n = Notifier([broken, lambda t, m: sent.append(m)], cooldown=0, clock=Clock())
    assert n.maybe_notify("t", "hello") is True
    assert sent == ["hello"]
