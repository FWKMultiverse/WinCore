import os
import time
from pathlib import Path

import pytest

from WinCore import atomic_write, AtomicWriteError


def test_basic_write(tmp_path: Path) -> None:
    dst = tmp_path / "out.txt"

    def _write(p: Path) -> None:
        with open(p, "w", encoding="utf-8") as f:
            f.write("hello")

    atomic_write(_write, dst)
    assert dst.read_text(encoding="utf-8") == "hello"
    # no leftover temp files
    assert list(tmp_path.glob(".*")) == []


def test_overwrite_existing(tmp_path: Path) -> None:
    dst = tmp_path / "out.txt"
    dst.write_text("old", encoding="utf-8")

    def _write(p: Path) -> None:
        with open(p, "w", encoding="utf-8") as f:
            f.write("new")

    atomic_write(_write, dst)
    assert dst.read_text(encoding="utf-8") == "new"


def test_write_fn_failure_leaves_destination_untouched(tmp_path: Path) -> None:
    dst = tmp_path / "out.txt"
    dst.write_text("original", encoding="utf-8")

    def _write(p: Path) -> None:
        raise ValueError("boom")

    with pytest.raises(ValueError):
        atomic_write(_write, dst)

    # destination untouched, no leftover temp file
    assert dst.read_text(encoding="utf-8") == "original"
    assert list(tmp_path.glob(".*tmp*")) == []


def test_retries_on_transient_replace_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dst = tmp_path / "out.txt"
    calls = {"n": 0}
    real_replace = os.replace

    def flaky_replace(src, dst_):  # noqa: ANN001
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError("simulated lock")
        return real_replace(src, dst_)

    monkeypatch.setattr(os, "replace", flaky_replace)
    monkeypatch.setattr(time, "sleep", lambda _: None)  # skip real backoff in tests

    def _write(p: Path) -> None:
        with open(p, "w", encoding="utf-8") as f:
            f.write("data")

    atomic_write(_write, dst, retries=5, initial_delay=0.01)
    assert dst.read_text(encoding="utf-8") == "data"
    assert calls["n"] == 3


def test_raises_after_exhausting_retries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dst = tmp_path / "out.txt"

    def always_locked(src, dst_):  # noqa: ANN001
        raise PermissionError("permanently locked")

    monkeypatch.setattr(os, "replace", always_locked)
    monkeypatch.setattr(time, "sleep", lambda _: None)

    def _write(p: Path) -> None:
        with open(p, "w", encoding="utf-8") as f:
            f.write("data")

    with pytest.raises(AtomicWriteError):
        atomic_write(_write, dst, retries=3, initial_delay=0.01)

    # temp file cleaned up, destination never created
    assert not dst.exists()
    assert list(tmp_path.glob(".*tmp*")) == []
