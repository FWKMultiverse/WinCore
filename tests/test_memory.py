from WinCore.memory import (
    WorkingSetTrimError,
    _decide_clear,
    _predict_future_fraction,
    estimate_worker_ram_multiplier,
    recommended_dataloader_kwargs,
    trim_working_set,
)


def test_windows_caps_workers_lower_than_raw_thread_count(monkeypatch):
    import WinCore.memory as memory

    monkeypatch.setattr(memory.platform, "system", lambda: "Windows")
    plan = recommended_dataloader_kwargs(cpu_recommended_threads=32, cuda_available=True)
    assert plan.num_workers <= 6
    assert plan.pin_memory is True
    assert plan.persistent_workers is True
    assert plan.prefetch_factor == 4


def test_linux_uses_full_recommended_threads(monkeypatch):
    import WinCore.memory as memory

    monkeypatch.setattr(memory.platform, "system", lambda: "Linux")
    plan = recommended_dataloader_kwargs(cpu_recommended_threads=12, cuda_available=False)
    assert plan.num_workers == 12
    assert plan.pin_memory is False


def test_zero_workers_disables_persistent_and_prefetch(monkeypatch):
    import WinCore.memory as memory

    monkeypatch.setattr(memory.platform, "system", lambda: "Linux")
    plan = recommended_dataloader_kwargs(cpu_recommended_threads=0, cuda_available=False)
    assert plan.num_workers == 0
    assert plan.persistent_workers is False
    assert plan.prefetch_factor is None


# --- CacheGuard(adaptive=True) predictive-clearing logic ---
# These test the pure decision functions directly, deliberately without
# torch/CUDA -- the trend math and clear/no-clear decision are ordinary
# Python and don't need a GPU to verify. CacheGuard.check() itself still
# requires real CUDA and is not covered here (there's no GPU in CI for
# this sandbox); what's covered is the actual logic bug surface: does it
# predict correctly, and does it clear at the right moments.

def test_predict_future_fraction_flat_history_no_trend():
    assert _predict_future_fraction([0.5, 0.5, 0.5, 0.5], lookahead=5) == 0.5


def test_predict_future_fraction_declining_trend_extrapolates_further_down():
    history = [0.30, 0.25, 0.20, 0.15]  # dropping ~0.05 per reading
    predicted = _predict_future_fraction(history, lookahead=2)
    assert predicted < history[-1]


def test_predict_future_fraction_rising_trend_extrapolates_further_up():
    history = [0.10, 0.15, 0.20, 0.25]
    predicted = _predict_future_fraction(history, lookahead=2)
    assert predicted > history[-1]


def test_predict_future_fraction_handles_short_history():
    assert _predict_future_fraction([], lookahead=3) == 1.0
    assert _predict_future_fraction([0.42], lookahead=3) == 0.42


def test_decide_clear_reactive_fires_regardless_of_adaptive_flag():
    # already below threshold -> clears immediately, non-predictive,
    # whether or not adaptive mode is on
    for adaptive in (False, True):
        should, predictive = _decide_clear(
            free_fraction=0.05, min_free_fraction=0.10,
            adaptive=adaptive, history=[], lookahead_checks=3,
        )
        assert should is True
        assert predictive is False


def test_decide_clear_non_adaptive_never_clears_early():
    declining = [0.30, 0.25, 0.20, 0.15]
    should, predictive = _decide_clear(
        free_fraction=0.20, min_free_fraction=0.10,
        adaptive=False, history=declining, lookahead_checks=2,
    )
    assert should is False
    assert predictive is False


def test_decide_clear_adaptive_fires_early_on_declining_trend():
    # still above min_free_fraction right now, but the trend predicts
    # crossing it within lookahead_checks -- adaptive mode should clear
    # NOW, before the hard threshold is actually crossed.
    declining = [0.30, 0.25, 0.20, 0.15]
    should, predictive = _decide_clear(
        free_fraction=0.20, min_free_fraction=0.10,
        adaptive=True, history=declining, lookahead_checks=2,
    )
    assert should is True
    assert predictive is True


def test_decide_clear_adaptive_does_not_fire_on_stable_history():
    stable = [0.40, 0.41, 0.40, 0.39]
    should, predictive = _decide_clear(
        free_fraction=0.40, min_free_fraction=0.10,
        adaptive=True, history=stable, lookahead_checks=3,
    )
    assert should is False
    assert predictive is False


def test_decide_clear_adaptive_requires_minimum_history_length():
    # only 2 readings so far -- not enough to trust a trend yet, even
    # though it looks like a sharp decline
    short_history = [0.30, 0.12]
    should, predictive = _decide_clear(
        free_fraction=0.15, min_free_fraction=0.10,
        adaptive=True, history=short_history, lookahead_checks=2,
        min_history_for_prediction=3,
    )
    assert should is False
    assert predictive is False


def test_trim_working_set_is_real_noop_on_non_windows(monkeypatch):
    import WinCore.memory as memory

    monkeypatch.setattr(memory.platform, "system", lambda: "Linux")
    assert trim_working_set() is False


def test_trim_working_set_calls_real_win32_api_on_windows(monkeypatch):
    """Doesn't run the actual Win32 call (this sandbox isn't Windows and
    `ctypes.windll` doesn't exist off-Windows) -- stubs `ctypes.windll`
    with a fake kernel32 to verify WinCore calls SetProcessWorkingSetSize
    with the documented (-1, -1) 'trim now' sentinel, not some other
    arguments."""
    import ctypes
    import types

    import WinCore.memory as memory

    monkeypatch.setattr(memory.platform, "system", lambda: "Windows")

    calls = []

    class FakeKernel32:
        def GetCurrentProcess(self):
            return 12345

        def SetProcessWorkingSetSize(self, handle, min_size, max_size):
            calls.append((handle, min_size, max_size))
            return 1  # nonzero = success, matches Win32 BOOL convention

    fake_windll = types.SimpleNamespace(kernel32=FakeKernel32())
    monkeypatch.setattr(ctypes, "windll", fake_windll, raising=False)

    result = trim_working_set()
    assert result is True
    assert calls == [(12345, -1, -1)]


def test_trim_working_set_raises_on_win32_failure(monkeypatch):
    import ctypes
    import types

    import WinCore.memory as memory

    monkeypatch.setattr(memory.platform, "system", lambda: "Windows")

    class FailingKernel32:
        def GetCurrentProcess(self):
            return 1

        def SetProcessWorkingSetSize(self, handle, min_size, max_size):
            return 0  # zero = failure, matches Win32 BOOL convention

    fake_windll = types.SimpleNamespace(kernel32=FailingKernel32())
    monkeypatch.setattr(ctypes, "windll", fake_windll, raising=False)
    monkeypatch.setattr(ctypes, "get_last_error", lambda: 5, raising=False)

    import pytest

    with pytest.raises(WorkingSetTrimError):
        trim_working_set()


def test_estimate_worker_ram_multiplier_zero_workers():
    msg = estimate_worker_ram_multiplier(0)
    assert "no per-worker" in msg


def test_estimate_worker_ram_multiplier_mentions_multiplier():
    msg = estimate_worker_ram_multiplier(4)
    assert "5x" in msg  # num_workers + 1
    assert "spawn" not in msg.lower() or "process" in msg.lower()
