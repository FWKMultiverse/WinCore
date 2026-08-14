"""
Atomic, retry-safe file writes.

Why this exists
----------------
POSIX (Linux/Mac) lets you replace a file that another process has open for
reading — unlink() just removes a directory entry, the open file handle
keeps working until it's closed. Windows enforces file locks much more
strictly: if antivirus, a search indexer, or a sync client (OneDrive,
Dropbox) has your target file open even briefly, `os.replace()` /
`open(path, "w")` can raise `PermissionError: [WinError 32]`. This happens
*randomly* — whenever the write happens to land in the same instant as the
external scan — which makes it maddening to debug from a single crash log.

This module fixes it with the standard two-part pattern:
  1. Write to a fresh, PID-suffixed temp file in the same directory as the
     destination (same directory matters: os.replace() must stay on one
     filesystem/volume to be atomic).
  2. os.replace() the temp file onto the destination, retrying with
     short exponential backoff if the OS reports the destination is
     locked.

On Linux/Mac this loop is a no-op in practice — the first replace()
attempt always succeeds, so there's no performance cost to using this
everywhere instead of only on Windows.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable, Optional, Union


class AtomicWriteError(OSError):
    """Raised when a write could not be committed after all retries."""


def atomic_write(
    write_fn: Callable[[Path], None],
    dst: Union[str, Path],
    retries: int = 6,
    initial_delay: float = 0.25,
    max_delay: float = 3.0,
) -> None:
    """Write a file atomically, with retry-on-lock for Windows.

    Args:
        write_fn: called once with a temp `Path` to write to. Must fully
            write and close the file (use a `with open(...) as f:` block
            inside — an unclosed handle will itself cause the retry loop
            below to fail, since Windows won't let you replace a file you
            still have open).
        dst: final destination path. Parent directory must already exist.
        retries: max attempts for the final `os.replace()` step.
        initial_delay: seconds to wait before the first retry.
        max_delay: cap for exponential backoff between retries.

    Raises:
        AtomicWriteError: if `os.replace()` still fails after all retries
            (destination is left untouched — only the temp file is lost).
        Exception: whatever `write_fn` itself raises, propagated directly
            (the destination is never touched in this case either).

    Example:
        >>> def _write(p):
        ...     with open(p, "w", encoding="utf-8") as f:
        ...         f.write("hello")
        >>> atomic_write(_write, "output.txt")
    """
    dst = Path(dst)
    tmp = dst.parent / f".{dst.name}.tmp{os.getpid()}"

    try:
        write_fn(tmp)
    except Exception:
        _silent_unlink(tmp)
        raise

    delay = initial_delay
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            os.replace(tmp, dst)
            return
        except (PermissionError, OSError) as e:
            last_err = e
            if attempt == retries - 1:
                break
            time.sleep(delay)
            delay = min(delay * 2, max_delay)

    _silent_unlink(tmp)
    raise AtomicWriteError(
        f"Could not write {dst} after {retries} attempts — the destination "
        f"is likely locked by another process (antivirus, search indexer, "
        f"or a sync client like OneDrive/Dropbox scanning it mid-write). "
        f"Try excluding this folder from real-time scanning, or pausing "
        f"sync temporarily: {last_err}"
    ) from last_err


def _silent_unlink(p: Path) -> None:
    try:
        if p.exists():
            p.unlink()
    except OSError:
        pass


def atomic_torch_save(
    obj: Any,
    dst: Union[str, Path],
    retries: int = 6,
    initial_delay: float = 0.25,
    max_delay: float = 3.0,
    **torch_save_kwargs: Any,
) -> None:
    """Save `obj` via `torch.save()` with the same atomic-write/retry
    guarantee as `atomic_write` -- convenience wrapper so you don't have
    to write the `lambda p: torch.save(obj, p)` yourself every time.

    `obj` can be anything `torch.save` accepts (a state_dict, a full
    checkpoint dict with optimizer/scheduler state, etc.) -- this
    doesn't change what gets saved, only how safely it lands on disk.
    `**torch_save_kwargs` is forwarded straight to `torch.save` (e.g.
    `pickle_protocol=...`).

    `torch` is imported lazily inside this function, not at module
    level, so `WinCore.io` itself keeps its zero-hard-dependency
    guarantee -- this specific function just needs torch installed to
    be called, same as the rest of WinCore's torch-touching functions.

    Example:
        WinCore.io.atomic_torch_save(model.state_dict(), "checkpoint.pt")
    """
    import torch

    atomic_write(
        lambda p: torch.save(obj, p, **torch_save_kwargs),
        dst,
        retries=retries,
        initial_delay=initial_delay,
        max_delay=max_delay,
    )


def atomic_safetensors_save(
    tensors: dict,
    dst: Union[str, Path],
    metadata: Optional[dict] = None,
    retries: int = 6,
    initial_delay: float = 0.25,
    max_delay: float = 3.0,
) -> None:
    """Save `tensors` via `safetensors.torch.save_file()` with the same
    atomic-write/retry guarantee as `atomic_write`.

    `tensors` must be a flat `dict[str, torch.Tensor]` -- that's
    safetensors' own format requirement (unlike `torch.save`, it can't
    serialize arbitrary Python objects/nested structures by design,
    which is exactly why it's a safer format to load from an untrusted
    source: no pickle involved). `metadata`, if given, is a
    `dict[str, str]` stored alongside the tensors (safetensors requires
    string values specifically -- not arbitrary JSON).

    `safetensors` is an OPTIONAL dependency (see the `safetensors` extra
    in `pyproject.toml`) -- this raises a clear `ImportError` naming the
    exact `pip install` line if it isn't installed, rather than a
    confusing `AttributeError` from deep inside some other code path.

    Example:
        WinCore.io.atomic_safetensors_save(
            model.state_dict(), "checkpoint.safetensors",
            metadata={"format": "pt"},
        )
    """
    try:
        from safetensors.torch import save_file
    except ImportError as e:
        raise ImportError(
            "atomic_safetensors_save needs the optional 'safetensors' "
            "package, which isn't installed. Install it with:\n\n"
            "    pip install safetensors\n\n"
            "(or `pip install \"WinCore[safetensors]\"` if using that extra)"
        ) from e

    atomic_write(
        lambda p: save_file(tensors, str(p), metadata=metadata),
        dst,
        retries=retries,
        initial_delay=initial_delay,
        max_delay=max_delay,
    )
