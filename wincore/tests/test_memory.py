from WinCore.memory import recommended_dataloader_kwargs, _predict_future_fraction, _decide_clear


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
