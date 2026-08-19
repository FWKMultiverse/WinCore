"""
Tests for WinCore.memory.PinnedBufferPool -- reuse of pinned CPU
buffers by (shape, dtype). torch is faked via sys.modules (same
convention as test_io_fast_load.py / test_precision_cuda_perf.py)
since this only needs to exercise the pooling logic, not real CUDA
pinning.
"""
import sys
import types

import pytest

from WinCore.memory import PinnedBufferPool


class _FakeDtype:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"fake.{self.name}"


class _FakeTensor:
    def __init__(self, shape, dtype):
        self.shape = tuple(shape)
        self.dtype = dtype


@pytest.fixture
def fake_torch(monkeypatch):
    calls = []
    mod = types.ModuleType("torch")
    mod.float32 = _FakeDtype("float32")
    mod.float16 = _FakeDtype("float16")

    def fake_empty(shape, dtype=None, pin_memory=False):
        calls.append({"shape": tuple(shape), "dtype": dtype, "pin_memory": pin_memory})
        return _FakeTensor(shape, dtype)

    mod.empty = fake_empty
    monkeypatch.setitem(sys.modules, "torch", mod)
    return calls, mod


def test_get_allocates_new_buffer_on_first_call(fake_torch):
    calls, mod = fake_torch
    pool = PinnedBufferPool()
    buf = pool.get((4, 4), mod.float32)
    assert buf.shape == (4, 4)
    assert buf.dtype is mod.float32
    assert len(calls) == 1
    assert calls[0]["pin_memory"] is True


def test_get_defaults_dtype_to_float32(fake_torch):
    calls, mod = fake_torch
    pool = PinnedBufferPool()
    pool.get((2, 2))
    assert calls[0]["dtype"] is mod.float32


def test_release_then_get_reuses_same_object_no_new_allocation(fake_torch):
    calls, mod = fake_torch
    pool = PinnedBufferPool()
    buf1 = pool.get((8, 8), mod.float32)
    pool.release(buf1)
    assert len(calls) == 1

    buf2 = pool.get((8, 8), mod.float32)
    assert buf2 is buf1  # exact same object, not a new allocation
    assert len(calls) == 1  # no second torch.empty call


def test_different_shape_does_not_reuse(fake_torch):
    calls, mod = fake_torch
    pool = PinnedBufferPool()
    buf1 = pool.get((4, 4), mod.float32)
    pool.release(buf1)

    buf2 = pool.get((8, 8), mod.float32)
    assert buf2 is not buf1
    assert len(calls) == 2


def test_different_dtype_does_not_reuse(fake_torch):
    calls, mod = fake_torch
    pool = PinnedBufferPool()
    buf1 = pool.get((4, 4), mod.float32)
    pool.release(buf1)

    buf2 = pool.get((4, 4), mod.float16)
    assert buf2 is not buf1
    assert len(calls) == 2


def test_len_reflects_pooled_buffer_count(fake_torch):
    _, mod = fake_torch
    pool = PinnedBufferPool()
    assert len(pool) == 0
    buf = pool.get((4, 4), mod.float32)
    assert len(pool) == 0  # not released yet
    pool.release(buf)
    assert len(pool) == 1


def test_max_buffers_evicts_oldest_when_exceeded(fake_torch):
    _, mod = fake_torch
    pool = PinnedBufferPool(max_buffers=2)
    b1 = pool.get((1, 1), mod.float32)
    b2 = pool.get((2, 2), mod.float32)
    b3 = pool.get((3, 3), mod.float32)
    pool.release(b1)  # oldest
    pool.release(b2)
    pool.release(b3)  # pool now has 3 pending -> should evict b1's slot
    assert len(pool) == 2

    # b1's (1,1) shape should have been evicted -- a get() for it
    # allocates fresh again instead of reusing.
    calls_before = []
    mod.empty = lambda shape, dtype=None, pin_memory=False: (
        calls_before.append(1) or _FakeTensor(shape, dtype)
    )
    pool.get((1, 1), mod.float32)
    assert len(calls_before) == 1  # had to allocate -- (1,1) was evicted


def test_release_using_tensor_own_shape_dtype_not_explicit_args(fake_torch):
    """release() reads shape/dtype off the tensor itself, not from
    separate arguments -- so a caller doesn't need to remember what
    they originally asked get() for."""
    _, mod = fake_torch
    pool = PinnedBufferPool()
    buf = pool.get((5, 5), mod.float32)
    pool.release(buf)
    again = pool.get((5, 5), mod.float32)
    assert again is buf
