"""
Tests for WinCore.io.fast_torch_load / fast_safetensors_load /
load_checkpoint -- the read-side counterparts to
atomic_torch_save / atomic_safetensors_save (see
test_io_checkpoint_formats.py for the save side).

Same convention as that file: torch and safetensors are monkeypatched
via sys.modules rather than requiring either to actually be installed,
since WinCore.io imports them lazily inside each function body.
"""
import sys
import types
from pathlib import Path

import pytest

from WinCore.io import fast_torch_load, fast_safetensors_load, load_checkpoint


@pytest.fixture
def fake_torch_mmap_capable(monkeypatch):
    """A fake torch.load that accepts mmap= and records how it was called."""
    calls = []
    mod = types.ModuleType("torch")

    def fake_load(src, mmap=False, **kwargs):
        calls.append({"src": str(src), "mmap": mmap, "kwargs": kwargs})
        return {"state": "loaded", "via_mmap": mmap}

    mod.load = fake_load
    monkeypatch.setitem(sys.modules, "torch", mod)
    return calls


@pytest.fixture
def fake_torch_old_no_mmap_kwarg(monkeypatch):
    """Simulates torch < 2.1: torch.load() raises TypeError on mmap=."""
    calls = []
    mod = types.ModuleType("torch")

    def fake_load(src, **kwargs):
        if "mmap" in kwargs:
            raise TypeError("load() got an unexpected keyword argument 'mmap'")
        calls.append({"src": str(src), "kwargs": kwargs})
        return {"state": "loaded-classic"}

    mod.load = fake_load
    monkeypatch.setitem(sys.modules, "torch", mod)
    return calls


@pytest.fixture
def fake_torch_mmap_runtime_failure(monkeypatch):
    """Simulates mmap being rejected for this specific file/filesystem
    (e.g. a network share) -- RuntimeError on mmap=True, succeeds without it."""
    calls = []
    mod = types.ModuleType("torch")

    def fake_load(src, mmap=False, **kwargs):
        if mmap:
            raise RuntimeError("mmap not supported on this filesystem")
        calls.append({"src": str(src), "kwargs": kwargs})
        return {"state": "loaded-fallback"}

    mod.load = fake_load
    monkeypatch.setitem(sys.modules, "torch", mod)
    return calls


@pytest.fixture
def fake_safetensors(monkeypatch):
    calls = []
    st_mod = types.ModuleType("safetensors")
    st_torch_mod = types.ModuleType("safetensors.torch")

    def fake_load_file(path, device=None):
        calls.append({"path": path, "device": device})
        return {"w": "fake-tensor"}

    st_torch_mod.load_file = fake_load_file
    st_mod.torch = st_torch_mod
    monkeypatch.setitem(sys.modules, "safetensors", st_mod)
    monkeypatch.setitem(sys.modules, "safetensors.torch", st_torch_mod)
    return calls


# -- fast_torch_load ---------------------------------------------------


def test_fast_torch_load_uses_mmap_by_default(tmp_path: Path, fake_torch_mmap_capable):
    dst = tmp_path / "checkpoint.pt"
    result = fast_torch_load(dst)
    assert result == {"state": "loaded", "via_mmap": True}
    assert fake_torch_mmap_capable[0]["mmap"] is True


def test_fast_torch_load_forwards_device_as_map_location(tmp_path: Path, fake_torch_mmap_capable):
    dst = tmp_path / "checkpoint.pt"
    fast_torch_load(dst, device="cuda:0")
    assert fake_torch_mmap_capable[0]["kwargs"]["map_location"] == "cuda:0"


def test_fast_torch_load_mmap_false_skips_mmap_path(tmp_path: Path, fake_torch_mmap_capable):
    dst = tmp_path / "checkpoint.pt"
    result = fast_torch_load(dst, mmap=False)
    assert result == {"state": "loaded", "via_mmap": False}


def test_fast_torch_load_falls_back_on_old_torch_without_mmap_kwarg(
    tmp_path: Path, fake_torch_old_no_mmap_kwarg
):
    dst = tmp_path / "checkpoint.pt"
    result = fast_torch_load(dst)
    assert result == {"state": "loaded-classic"}
    assert len(fake_torch_old_no_mmap_kwarg) == 1


def test_fast_torch_load_falls_back_on_mmap_runtime_failure(
    tmp_path: Path, fake_torch_mmap_runtime_failure
):
    dst = tmp_path / "checkpoint.pt"
    result = fast_torch_load(dst)
    assert result == {"state": "loaded-fallback"}
    assert len(fake_torch_mmap_runtime_failure) == 1


def test_fast_torch_load_weights_only_forwarded_only_when_given(
    tmp_path: Path, fake_torch_mmap_capable
):
    dst = tmp_path / "checkpoint.pt"
    fast_torch_load(dst, weights_only=True)
    assert fake_torch_mmap_capable[0]["kwargs"]["weights_only"] is True

    fake_torch_mmap_capable.clear()
    fast_torch_load(dst)
    assert "weights_only" not in fake_torch_mmap_capable[0]["kwargs"]


# -- fast_safetensors_load ----------------------------------------------


def test_fast_safetensors_load_default_device(tmp_path: Path, fake_safetensors):
    dst = tmp_path / "checkpoint.safetensors"
    result = fast_safetensors_load(dst)
    assert result == {"w": "fake-tensor"}
    assert fake_safetensors[0]["device"] is None


def test_fast_safetensors_load_forwards_device(tmp_path: Path, fake_safetensors):
    dst = tmp_path / "checkpoint.safetensors"
    fast_safetensors_load(dst, device="cuda:0")
    assert fake_safetensors[0]["device"] == "cuda:0"


def test_fast_safetensors_load_raises_clear_error_when_not_installed(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "safetensors", None)
    monkeypatch.setitem(sys.modules, "safetensors.torch", None)

    with pytest.raises(ImportError, match="pip install safetensors"):
        fast_safetensors_load(tmp_path / "out.safetensors")


# -- load_checkpoint (format dispatch) -----------------------------------


def test_load_checkpoint_dispatches_safetensors_by_extension(tmp_path: Path, fake_safetensors):
    dst = tmp_path / "checkpoint.safetensors"
    result = load_checkpoint(dst, device="cuda:0")
    assert result == {"w": "fake-tensor"}
    assert fake_safetensors[0]["device"] == "cuda:0"


def test_load_checkpoint_dispatches_safetensors_case_insensitive(tmp_path: Path, fake_safetensors):
    dst = tmp_path / "checkpoint.SAFETENSORS"
    load_checkpoint(dst)
    assert len(fake_safetensors) == 1


def test_load_checkpoint_dispatches_torch_for_other_extensions(tmp_path: Path, fake_torch_mmap_capable):
    for suffix in (".pt", ".pth", ".bin", ".ckpt"):
        fake_torch_mmap_capable.clear()
        dst = tmp_path / f"checkpoint{suffix}"
        result = load_checkpoint(dst)
        assert result == {"state": "loaded", "via_mmap": True}
        assert len(fake_torch_mmap_capable) == 1


def test_load_checkpoint_mmap_flag_only_affects_torch_path(tmp_path: Path, fake_torch_mmap_capable):
    dst = tmp_path / "checkpoint.pt"
    load_checkpoint(dst, mmap=False)
    assert fake_torch_mmap_capable[0]["mmap"] is False
