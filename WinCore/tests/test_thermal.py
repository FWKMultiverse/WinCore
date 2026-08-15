"""
Tests for WinCore.thermal.ThermalGuard -- graduated (escalating, then
resetting) pause behavior and the critical_threshold_c/on_critical
hook. All GPU access is monkeypatched (get_gpu_temperature and
time.sleep), so this runs instantly with no real GPU/hardware needed.
"""
import pytest

import WinCore.thermal as thermal


def _guard_with_temp_sequence(monkeypatch, temps, **kwargs):
    it = iter(temps)
    monkeypatch.setattr(thermal, "get_gpu_temperature", lambda idx=0: next(it, None))
    sleeps = []
    monkeypatch.setattr(thermal.time, "sleep", lambda s: sleeps.append(s))
    guard = thermal.ThermalGuard(**kwargs)
    return guard, sleeps


def test_no_pause_under_threshold(monkeypatch):
    guard, sleeps = _guard_with_temp_sequence(monkeypatch, [70, 75, 80], threshold_c=83)
    for _ in range(3):
        assert guard.check() is None
    assert sleeps == []


def test_pause_escalates_geometrically_while_consecutively_over_threshold(monkeypatch):
    guard, sleeps = _guard_with_temp_sequence(
        monkeypatch, [85, 86, 87, 88],
        threshold_c=83, pause_seconds=2.0, backoff_factor=1.5, max_pause_seconds=100.0,
    )
    events = [guard.check() for _ in range(4)]
    paused = [round(e.paused_seconds, 4) for e in events]
    assert paused == [2.0, 3.0, 4.5, 6.75]


def test_pause_capped_at_max_pause_seconds(monkeypatch):
    guard, sleeps = _guard_with_temp_sequence(
        monkeypatch, [85] * 10,
        threshold_c=83, pause_seconds=2.0, backoff_factor=2.0, max_pause_seconds=10.0,
    )
    events = [guard.check() for _ in range(10)]
    assert all(e.paused_seconds <= 10.0 for e in events)
    assert events[-1].paused_seconds == 10.0  # eventually pinned at the cap


def test_pause_resets_after_a_cool_reading(monkeypatch):
    guard, sleeps = _guard_with_temp_sequence(
        monkeypatch, [85, 86, 70, 85],
        threshold_c=83, pause_seconds=2.0, backoff_factor=1.5, max_pause_seconds=100.0,
    )
    e1 = guard.check()
    e2 = guard.check()
    e3 = guard.check()  # cool reading -> None, resets counter
    e4 = guard.check()  # hot again -> back to base pause, not escalated further
    assert round(e1.paused_seconds, 4) == 2.0
    assert round(e2.paused_seconds, 4) == 3.0
    assert e3 is None
    assert round(e4.paused_seconds, 4) == 2.0


def test_on_critical_fires_before_sleep_only_at_or_above_critical_threshold(monkeypatch):
    critical_events = []
    order = []
    guard, sleeps = _guard_with_temp_sequence(
        monkeypatch, [85, 91],
        threshold_c=83, critical_threshold_c=90,
        on_critical=lambda e: (critical_events.append(e), order.append("critical"))[0],
    )
    # wrap sleep to also record ordering relative to on_critical
    real_sleep = thermal.time.sleep
    monkeypatch.setattr(thermal.time, "sleep", lambda s: (order.append("sleep"), real_sleep(s)))

    e1 = guard.check()  # 85C: over threshold but not critical
    assert e1.critical is False
    assert critical_events == []
    order.clear()  # e1's own (non-critical) pause also sleeps -- isolate e2's ordering

    e2 = guard.check()  # 91C: critical
    assert e2.critical is True
    assert len(critical_events) == 1
    assert critical_events[0].temperature_c == 91
    # on_critical must run BEFORE the sleep for that same check
    assert order == ["critical", "sleep"]


def test_no_critical_threshold_set_never_fires_on_critical(monkeypatch):
    calls = []
    guard, sleeps = _guard_with_temp_sequence(
        monkeypatch, [99],  # very hot, but no critical_threshold_c configured
        threshold_c=83, on_critical=lambda e: calls.append(e),
    )
    event = guard.check()
    assert event.critical is False
    assert calls == []


def test_returns_none_and_resets_when_temperature_unreadable(monkeypatch):
    guard, sleeps = _guard_with_temp_sequence(monkeypatch, [85, None, 85], threshold_c=83, pause_seconds=2.0)
    e1 = guard.check()
    e2 = guard.check()  # unreadable -> None, resets streak
    e3 = guard.check()
    assert round(e1.paused_seconds, 4) == 2.0
    assert e2 is None
    assert round(e3.paused_seconds, 4) == 2.0  # NOT escalated past e2
