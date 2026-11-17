import pytest

from modelguard.alerts import AlertEngine
from modelguard.monitors import MonitorResult


def r(flagged, monitor="psi", feature="age"):
    return MonitorResult(monitor, feature, 0.5, None, flagged)


def test_alerts_exactly_at_n_and_once_per_streak():
    e = AlertEngine(3)
    got = [e.update(i, [r(True)]) for i in range(6)]
    assert [len(g) for g in got] == [0, 0, 1, 0, 0, 0]
    a = got[2][0]
    assert a.batch_index == 2 and a.feature == "age" and "3 consecutive" in a.reason


def test_clear_batch_resets_streak_and_rearms():
    e = AlertEngine(3)
    seq = [True, True, False, True, True, True, False, True, True, True]
    assert [len(e.update(i, [r(f)])) for i, f in enumerate(seq)] == [0, 0, 0, 0, 0, 1, 0, 0, 0, 1]


def test_absent_monitor_counts_as_clear():
    e = AlertEngine(2)
    e.update(0, [r(True)])
    e.update(1, [])
    assert e.update(2, [r(True)]) == []


def test_streaks_are_per_monitor_and_feature():
    e = AlertEngine(2)
    e.update(0, [r(True, feature="age"), r(True, feature="sex")])
    out = e.update(1, [r(True, feature="age")])
    assert [a.feature for a in out] == ["age"]


def test_invalid_n():
    with pytest.raises(ValueError):
        AlertEngine(0)
