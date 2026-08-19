"""
Tests for WinCore.accumulate.GradientAccumulator -- loss scaling,
boundary detection, and DDP no_sync() context selection.

No torch dependency needed: scale_loss() just divides whatever object
is passed (a plain float/int works fine as a stand-in for a loss
tensor for arithmetic purposes), and sync_context() only needs a
duck-typed fake "model" with a no_sync() method.
"""
import contextlib

import pytest

from WinCore.accumulate import GradientAccumulator


class _FakeNoSyncContext:
    """Records whether it was entered, for assertions."""

    def __init__(self, log, label):
        self._log = log
        self._label = label

    def __enter__(self):
        self._log.append(self._label)
        return self

    def __exit__(self, *exc):
        return False


class _FakeDDPModel:
    def __init__(self):
        self.log = []

    def no_sync(self):
        return _FakeNoSyncContext(self.log, "no_sync")


# -- construction -------------------------------------------------------


def test_rejects_zero_or_negative_accumulation_steps():
    with pytest.raises(ValueError):
        GradientAccumulator(accumulation_steps=0)
    with pytest.raises(ValueError):
        GradientAccumulator(accumulation_steps=-1)


def test_rejects_non_integer_accumulation_steps():
    """Regression test: a non-integer accumulation_steps (e.g. from
    total_batch_size / micro_batch_size not landing on a whole number)
    used to be silently accepted, producing an inconsistent window
    size (e.g. 2.5 alternates between 3-step and 2-step windows)
    instead of a fixed N -- directly undermining the loss-scaling
    correctness guarantee this class exists for."""
    for bad in (2.5, "4", 4.0, None, [4]):
        with pytest.raises(TypeError):
            GradientAccumulator(accumulation_steps=bad)


def test_accumulation_steps_one_is_valid():
    accum = GradientAccumulator(accumulation_steps=1)
    assert accum.step() is True  # every step is immediately a boundary


# -- scale_loss -----------------------------------------------------------


def test_scale_loss_divides_by_accumulation_steps():
    accum = GradientAccumulator(accumulation_steps=4)
    assert accum.scale_loss(8.0) == 2.0


def test_scale_loss_with_accumulation_steps_one_is_unchanged():
    accum = GradientAccumulator(accumulation_steps=1)
    assert accum.scale_loss(5.0) == 5.0


# -- step() / is_boundary() boundary sequencing ----------------------------


def test_step_returns_true_only_on_last_micro_step():
    accum = GradientAccumulator(accumulation_steps=4)
    results = [accum.step() for _ in range(4)]
    assert results == [False, False, False, True]


def test_step_wraps_to_new_window_after_boundary():
    accum = GradientAccumulator(accumulation_steps=2)
    results = [accum.step() for _ in range(6)]  # 3 full windows
    assert results == [False, True, False, True, False, True]


def test_is_boundary_true_only_right_before_last_micro_step():
    accum = GradientAccumulator(accumulation_steps=3)
    seen = []
    for _ in range(3):
        seen.append(accum.is_boundary())
        accum.step()
    assert seen == [False, False, True]


def test_reset_returns_to_start_of_window():
    accum = GradientAccumulator(accumulation_steps=4)
    accum.step()
    accum.step()
    accum.reset()
    assert accum.is_boundary() is False
    results = [accum.step() for _ in range(4)]
    assert results == [False, False, False, True]  # fresh window, not a partial one


# -- sync_context(): no model / no DDP --------------------------------------


def test_sync_context_is_nullcontext_without_model():
    accum = GradientAccumulator(accumulation_steps=4)
    ctx = accum.sync_context()
    assert isinstance(ctx, contextlib.nullcontext) or ctx.__class__ is type(contextlib.nullcontext())


def test_sync_context_is_nullcontext_when_model_has_no_no_sync():
    class PlainModel:
        pass

    accum = GradientAccumulator(accumulation_steps=4, model=PlainModel())
    with accum.sync_context():
        pass  # must not raise / must not try to call a missing no_sync()


# -- sync_context(): DDP model with no_sync() --------------------------------


def test_sync_context_uses_no_sync_on_non_boundary_steps():
    model = _FakeDDPModel()
    accum = GradientAccumulator(accumulation_steps=4, model=model)

    with accum.sync_context():
        pass
    accum.step()

    assert model.log == ["no_sync"]


def test_sync_context_uses_real_sync_on_boundary_step():
    model = _FakeDDPModel()
    accum = GradientAccumulator(accumulation_steps=2, model=model)

    # micro-step 1: non-boundary -> no_sync
    with accum.sync_context():
        pass
    accum.step()

    # micro-step 2: boundary -> real sync (nullcontext, not logged as no_sync)
    ctx = accum.sync_context()
    with ctx:
        pass
    accum.step()

    assert model.log == ["no_sync"]  # only ONE no_sync call for the whole window


def test_full_window_has_exactly_one_synced_step(monkeypatch):
    """End-to-end: across a full accumulation_steps=4 window, exactly
    1 of the 4 micro-steps should run under a real sync, and 3 should
    be skipped via no_sync() -- the actual communication-savings claim
    this module exists for."""
    model = _FakeDDPModel()
    accum = GradientAccumulator(accumulation_steps=4, model=model)

    for _ in range(4):
        with accum.sync_context():
            pass
        accum.step()

    assert model.log.count("no_sync") == 3  # micro-steps 1-3
    # the 4th (boundary) micro-step used nullcontext, which doesn't log
    # anything into model.log at all -- confirmed by the count above
    # being exactly 3, not 4.


def test_multiple_windows_each_get_exactly_one_sync():
    model = _FakeDDPModel()
    accum = GradientAccumulator(accumulation_steps=3, model=model)

    for _ in range(9):  # 3 full windows of 3
        with accum.sync_context():
            pass
        accum.step()

    assert model.log.count("no_sync") == 6  # 2 no_sync per window x 3 windows
