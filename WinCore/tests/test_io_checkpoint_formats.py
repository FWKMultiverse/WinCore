"""
Tests for WinCore.io.atomic_torch_save / atomic_safetensors_save.

Both are thin convenience wrappers around the already-tested
atomic_write (see test_io.py for the atomicity/retry/lock-safety
coverage itself) -- what's specific to test here is: (1) they call the
right underlying save function with the right arguments, and (2) the
safetensors wrapper's dependency handling. torch and safetensors are
both monkeypatched via sys.modules rather than requiring either to
actually be installed, since WinCore.io imports them lazily inside the
function body, not at module import time.
"""
import sys
import types
from pathlib import Path

import pytest

from WinCore.io import atomic_torch_save, atomic_safetensors_save


@pytest.fixture
def fake_torch(monkeypatch):
    calls = []
    mod = types.ModuleType("torch")

    def fake_save(obj, p, **kwargs):
        calls.append({"obj": obj, "path": str(p), "kwargs": kwargs})
        with open(p, "w", encoding="utf-8") as f:
            f.write("fake-torch-checkpoint")

    mod.save = fake_save
    monkeypatch.setitem(sys.modules, "torch", mod)
    return calls


@pytest.fixture
def fake_safetensors(monkeypatch):
    calls = []
    st_mod = types.ModuleType("safetensors")
    st_torch_mod = types.ModuleType("safetensors.torch")

    def fake_save_file(tensors, path, metadata=None):
        calls.append({"tensors": tensors, "path": path, "metadata": metadata})
        with open(path, "w", encoding="utf-8") as f:
            f.write("fake-safetensors-checkpoint")

    st_torch_mod.save_file = fake_save_file
    st_mod.torch = st_torch_mod
    monkeypatch.setitem(sys.modules, "safetensors", st_mod)
    monkeypatch.setitem(sys.modules, "safetensors.torch", st_torch_mod)
    return calls


def test_atomic_torch_save_writes_file_and_forwards_kwargs(tmp_path: Path, fake_torch):
    dst = tmp_path / "checkpoint.pt"
    state = {"weight": [1, 2, 3]}

    atomic_torch_save(state, dst, pickle_protocol=4)

    assert dst.exists()
    assert fake_torch[0]["obj"] is state
    assert fake_torch[0]["kwargs"] == {"pickle_protocol": 4}
    # no leftover temp file -- same atomicity guarantee as atomic_write
    assert list(tmp_path.glob(".*tmp*")) == []


def test_atomic_safetensors_save_writes_file_and_forwards_metadata(tmp_path: Path, fake_safetensors):
    dst = tmp_path / "checkpoint.safetensors"
    tensors = {"w": "fake-tensor-placeholder"}

    atomic_safetensors_save(tensors, dst, metadata={"format": "pt"})

    assert dst.exists()
    assert fake_safetensors[0]["tensors"] is tensors
    assert fake_safetensors[0]["metadata"] == {"format": "pt"}


def test_atomic_safetensors_save_raises_clear_error_when_not_installed(tmp_path, monkeypatch):
    # simulate safetensors genuinely not being installed
    monkeypatch.setitem(sys.modules, "safetensors", None)
    monkeypatch.setitem(sys.modules, "safetensors.torch", None)

    with pytest.raises(ImportError, match="pip install safetensors"):
        atomic_safetensors_save({"w": "x"}, tmp_path / "out.safetensors")
