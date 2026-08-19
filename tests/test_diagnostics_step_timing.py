"""
Tests for WinCore.diagnostics.TrainingMonitor.record_step_time() --
phase classification (warmup / steady_state / stalled) and ETA.

time.perf_counter is monkeypatched to a controllable fake clock so
step durations are exact and deterministic instead of relying on real
sleep() calls / wall-clock noise.
"""
import WinCore.diagnostics as diag_mod
from WinCore.diagnostics import TrainingMonitor, _default_warmup_steps


class _FakeClock:
    def __init__(self):
        self.t = 0.0

    def advance(self, seconds):
        self.t += seconds

    def __call__(self):
        return self.t


def _install_fake_clock(monkeypatch):
    clock = _FakeClock()
    monkeypatch.setattr(diag_mod.time, "perf_counter", clock)
    return clock


# -- _default_warmup_steps heuristic -------------------------------------


def test_default_warmup_steps_none_param_count():
    assert _default_warmup_steps(None) == 10


def test_default_warmup_steps_small_model():
    assert _default_warmup_steps(1_000_000) == 5


def test_default_warmup_steps_mid_model():
    assert _default_warmup_steps(300_000_000) == 15


def test_default_warmup_steps_large_model():
    assert _default_warmup_steps(7_000_000_000) == 30


def test_explicit_warmup_steps_overrides_heuristic():
    m = TrainingMonitor(warmup_steps=2, expected_param_count=7_000_000_000)
    assert m.warmup_steps == 2  # not 30 -- explicit wins


# -- phase classification -------------------------------------------------


def test_first_n_steps_are_warmup(monkeypatch):
    clock = _install_fake_clock(monkeypatch)
    m = TrainingMonitor(warmup_steps=3)

    statuses = []
    for i in range(3):
        clock.advance(1.0)
        statuses.append(m.record_step_time(step=i))

    assert [s.phase for s in statuses] == ["warmup", "warmup", "warmup"]
    assert all(s.steady_state_avg_seconds is None for s in statuses)


def test_steps_after_warmup_are_steady_state(monkeypatch):
    clock = _install_fake_clock(monkeypatch)
    m = TrainingMonitor(warmup_steps=2, stall_min_samples=2)

    for i in range(2):
        clock.advance(1.0)
        m.record_step_time(step=i)

    clock.advance(0.5)
    status = m.record_step_time(step=2)
    assert status.phase == "steady_state"
    assert status.steady_state_avg_seconds == 0.5


def test_stall_detected_after_min_samples(monkeypatch):
    clock = _install_fake_clock(monkeypatch)
    m = TrainingMonitor(warmup_steps=1, stall_min_samples=3, stall_factor=3.0)

    clock.advance(1.0)
    m.record_step_time(step=0)  # warmup

    # 3 steady-state steps at 0.5s each to establish a baseline
    for i in range(1, 4):
        clock.advance(0.5)
        status = m.record_step_time(step=i)
        assert status.phase == "steady_state"

    # a step that takes 5s (10x the 0.5s average, > stall_factor=3.0)
    clock.advance(5.0)
    stall_status = m.record_step_time(step=4)
    assert stall_status.phase == "stalled"
    assert stall_status.last_step_seconds == 5.0

    # stalled step's issue was emitted
    stall_issues = [iss for iss in m.issues if iss.code == "step_stall"]
    assert len(stall_issues) == 1
    assert stall_issues[0].step == 4
    assert stall_issues[0].severity == "warning"


def test_stall_not_folded_into_steady_state_average(monkeypatch):
    clock = _install_fake_clock(monkeypatch)
    m = TrainingMonitor(warmup_steps=1, stall_min_samples=3, stall_factor=3.0)

    clock.advance(1.0)
    m.record_step_time(step=0)
    for i in range(1, 4):
        clock.advance(0.5)
        m.record_step_time(step=i)

    clock.advance(5.0)
    m.record_step_time(step=4)  # stalled -- must not pollute the average

    clock.advance(0.5)
    status = m.record_step_time(step=5)
    assert status.phase == "steady_state"
    assert status.steady_state_avg_seconds == 0.5  # unchanged by the 5.0s stall


def test_no_stall_before_min_samples_reached(monkeypatch):
    clock = _install_fake_clock(monkeypatch)
    # stall_min_samples=5 but only 1 steady-state sample exists when a
    # slow step arrives -- must not be misclassified as a stall yet.
    m = TrainingMonitor(warmup_steps=1, stall_min_samples=5, stall_factor=3.0)

    clock.advance(1.0)
    m.record_step_time(step=0)  # warmup
    clock.advance(0.5)
    m.record_step_time(step=1)  # steady state, avg=0.5, only 1 sample

    clock.advance(5.0)  # would be a 10x outlier, but not enough baseline yet
    status = m.record_step_time(step=2)
    assert status.phase == "steady_state"


# -- ETA --------------------------------------------------------------------


def test_eta_none_without_total_steps(monkeypatch):
    clock = _install_fake_clock(monkeypatch)
    m = TrainingMonitor(warmup_steps=1)
    clock.advance(1.0)
    m.record_step_time(step=0)
    clock.advance(0.5)
    status = m.record_step_time(step=1)
    assert status.eta_seconds is None


def test_eta_none_during_warmup(monkeypatch):
    clock = _install_fake_clock(monkeypatch)
    m = TrainingMonitor(warmup_steps=5)
    clock.advance(1.0)
    status = m.record_step_time(step=0, total_steps=100)
    assert status.phase == "warmup"
    assert status.eta_seconds is None


def test_eta_computed_from_steady_state_average(monkeypatch):
    clock = _install_fake_clock(monkeypatch)
    m = TrainingMonitor(warmup_steps=1)
    clock.advance(1.0)
    m.record_step_time(step=0)  # warmup
    clock.advance(0.5)
    status = m.record_step_time(step=1, total_steps=11)
    # avg=0.5s/step, 10 steps remaining (11-1) -> 5.0s ETA
    assert status.steady_state_avg_seconds == 0.5
    assert status.eta_seconds == 5.0


def test_eta_zero_when_at_or_past_total_steps(monkeypatch):
    clock = _install_fake_clock(monkeypatch)
    m = TrainingMonitor(warmup_steps=1)
    clock.advance(1.0)
    m.record_step_time(step=0)
    clock.advance(0.5)
    status = m.record_step_time(step=20, total_steps=10)
    assert status.eta_seconds == 0.0


# -- steps_per_second / elapsed_seconds / steps_recorded --------------------


def test_steps_per_second_matches_average(monkeypatch):
    clock = _install_fake_clock(monkeypatch)
    m = TrainingMonitor(warmup_steps=1)
    clock.advance(1.0)
    m.record_step_time(step=0)
    clock.advance(0.25)
    status = m.record_step_time(step=1)
    assert status.steps_per_second == 4.0  # 1 / 0.25


def test_steps_per_second_none_before_steady_state(monkeypatch):
    clock = _install_fake_clock(monkeypatch)
    m = TrainingMonitor(warmup_steps=5)
    clock.advance(1.0)
    status = m.record_step_time(step=0)
    assert status.steps_per_second is None


def test_elapsed_seconds_accumulates_across_calls(monkeypatch):
    clock = _install_fake_clock(monkeypatch)
    m = TrainingMonitor(warmup_steps=1)
    clock.advance(1.0)
    m.record_step_time(step=0)  # this call sets the "start" reference point
    clock.advance(2.0)
    status = m.record_step_time(step=1)
    # elapsed_seconds is measured from the FIRST record_step_time() call,
    # not from clock zero -- so it's the 2.0s advance since that first
    # call, not the full 3.0s since the clock started.
    assert status.elapsed_seconds == 2.0


def test_steps_recorded_increments_every_call(monkeypatch):
    clock = _install_fake_clock(monkeypatch)
    m = TrainingMonitor(warmup_steps=2)
    for i in range(5):
        clock.advance(0.1)
        status = m.record_step_time(step=i)
    assert status.steps_recorded == 5
