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
import threading
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
    # PID alone isn't a unique temp name: two threads in the same
    # process calling atomic_write() for the same dst at the same
    # moment (e.g. a thread-pooled checkpoint helper) share a PID and
    # would collide on this filename. Thread id makes it unique per
    # (process, thread) instead of just per process.
    tmp = dst.parent / f".{dst.name}.tmp{os.getpid()}.{threading.get_ident()}"

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


# ---------------------------------------------------------------------------
# Loading -- the atomic_* functions above only cover the write side.
# Everything below answers the read side: "given a checkpoint already on
# disk, what's the fastest correct way to get it into (V)RAM on Windows?"
# ---------------------------------------------------------------------------


def fast_torch_load(
    src: Union[str, Path],
    device: Optional[Union[str, "object"]] = None,
    mmap: bool = True,
    weights_only: Optional[bool] = None,
    **torch_load_kwargs: Any,
):
    """Load a `torch.save`-format checkpoint (`.pt`/`.pth`/`.bin`) as
    fast as this machine allows, instead of the plain
    `torch.load(path, map_location=device)` most scripts default to.

    What "fast" means here, concretely
    -------------------------------------
    `torch.load`'s classic path is: read the whole file into a fresh
    heap buffer, unpickle it, then (if `map_location` targets CUDA)
    copy each tensor's storage again from that CPU buffer to VRAM --
    two full copies of the checkpoint's bytes before training can even
    start, on top of whatever the OS's own file-read already cost.
    Windows makes the read side of that worse than Linux/Mac tend to
    see: NTFS + Windows Defender real-time scanning + (for a checkpoint
    that happens to live in a synced folder) OneDrive/Dropbox's on-
    access hooks all sit in the read path, and none of that is
    something a training script can turn off from Python.

    `mmap=True` (default here; requires torch>=2.1, added specifically
    for this use case) memory-maps the file instead of eagerly reading
    it: tensor storages become views into the mapped pages, materialized
    lazily as each one is actually touched, and the OS page cache does
    the caching instead of Python allocating and copying a second
    buffer. Net effect: lower peak host RAM (you're not holding a full
    duplicate of the checkpoint in a Python bytes buffer while also
    unpickling it), and touching only the parameters you actually load
    -- e.g. `map_location=device` with mmap can, on torch's supported
    versions, stream storages more directly and skip re-reading pages
    you don't end up materializing. This doesn't change training math
    or checkpoint format at all -- same file, same bytes, same
    `state_dict()` you'd get from the classic path.

    Args:
        src: path to the checkpoint file.
        device: forwarded as `map_location` (e.g. `"cuda:0"`, a
            `torch.device`, or `"cpu"`). `None` (default) leaves
            `map_location` unset -- torch's own default (tensors land
            wherever they were saved from), same as calling
            `torch.load(src)` directly.
        mmap: try the mmap path first (default `True`). Falls back
            automatically, never raises just because mmap itself isn't
            available -- see "Fallback behavior" below.
        weights_only: forwarded to `torch.load` if given explicitly.
            Left as `None` (this function's own default) means "don't
            override torch's own default for the installed version" --
            newer torch defaults this to `True` for security (blocks
            arbitrary pickle globals); this function does not silently
            change that safety default one way or the other, only lets
            the caller pick it explicitly if they need to.
        **torch_load_kwargs: forwarded to `torch.load` verbatim
            (e.g. `pickle_module=...`).

    Fallback behavior (never raises just because the fast path failed)
    ---------------------------------------------------------------------
    mmap can legitimately fail even on torch>=2.1 -- most commonly on a
    network drive or a filesystem that doesn't support memory-mapping
    the way NTFS/ext4 do (some SMB/CIFS mounts, some virtual/overlay
    filesystems used by container or sync-client software). Rather
    than make the caller detect their own filesystem, this function
    tries mmap first and transparently retries with the classic
    (non-mmap) `torch.load` path if:
      - this torch build is older than 2.1 and doesn't accept the
        `mmap=` keyword at all (`TypeError`), or
      - the mmap attempt itself fails for a filesystem/OS reason
        (`RuntimeError`/`OSError`) -- same final result either way,
        just without the speed/memory win on this specific file.

    Returns exactly what `torch.load` returns -- a state_dict, or
    whatever object structure was originally passed to `torch.save`
    (this function does not interpret or restructure it).
    """
    import torch

    kwargs = dict(torch_load_kwargs)
    if device is not None:
        kwargs.setdefault("map_location", device)
    if weights_only is not None:
        kwargs.setdefault("weights_only", weights_only)

    if mmap:
        try:
            return torch.load(src, mmap=True, **kwargs)
        except TypeError:
            # This torch build doesn't have the mmap= parameter at all
            # (added in 2.1) -- fall through to the classic call below.
            pass
        except (RuntimeError, OSError):
            # mmap itself was rejected for this specific file/filesystem
            # (e.g. a network share) -- fall through and read normally
            # instead of failing the whole load over a speed optimization.
            pass

    return torch.load(src, **kwargs)


def fast_safetensors_load(src: Union[str, Path], device: Optional[str] = None) -> dict:
    """Load a `.safetensors` checkpoint, returned as a flat
    `dict[str, torch.Tensor]`.

    Unlike `fast_torch_load`, there is no separate "fast" code path to
    opt into here -- the safetensors format is mmap-based by design at
    the C++ layer (that's *why* it exists as a format: no pickle, and
    zero-copy reads from a memory-mapped file are the normal case, not
    an opt-in flag). This function's job is just giving that loader the
    same ergonomic, WinCore-consistent surface as `fast_torch_load` --
    a `device=` kwarg that loads tensors directly onto that device
    (e.g. `"cuda:0"`) rather than landing on CPU first and needing a
    separate `.to(device)` copy per tensor -- and the same clear,
    actionable `ImportError` as `atomic_safetensors_save` if the
    optional `safetensors` package isn't installed, instead of the
    caller hitting a bare `ModuleNotFoundError` from inside the format
    dispatch below.

    Args:
        src: path to the `.safetensors` file.
        device: `"cpu"`, `"cuda"`, `"cuda:0"`, etc. `None` (default)
            loads to CPU, matching `safetensors.torch.load_file`'s own
            default.

    Returns a flat `dict[str, torch.Tensor]` -- safetensors files are
    always a flat tensor map (no nested structures, no optimizer state,
    no arbitrary Python objects -- that restriction is what makes the
    format safe to load from an untrusted source in the first place).
    If you saved a full training checkpoint (optimizer/scheduler state
    alongside weights), that went through `atomic_torch_save` /
    `fast_torch_load` instead, not this format.
    """
    try:
        from safetensors.torch import load_file
    except ImportError as e:
        raise ImportError(
            "fast_safetensors_load needs the optional 'safetensors' "
            "package, which isn't installed. Install it with:\n\n"
            "    pip install safetensors\n\n"
            "(or `pip install \"WinCore[safetensors]\"` if using that extra)"
        ) from e

    if device is not None:
        return load_file(str(src), device=device)
    return load_file(str(src))


def load_checkpoint(
    src: Union[str, Path],
    device: Optional[Union[str, "object"]] = None,
    mmap: bool = True,
    **kwargs: Any,
):
    """Format-dispatching convenience wrapper: picks
    `fast_safetensors_load` or `fast_torch_load` based on `src`'s file
    extension, so calling code doesn't need its own if/else on
    `.safetensors` vs `.pt`/`.pth`/`.bin`/`.ckpt` when it wants "just
    load this checkpoint as fast as possible, whichever format it's
    in" -- e.g. a training script that supports resuming from either
    format a run happened to be saved in.

    Args:
        src: checkpoint path. `.safetensors` (any case) dispatches to
            `fast_safetensors_load`; anything else dispatches to
            `fast_torch_load`.
        device: forwarded as `device=` (safetensors path) or
            `map_location=` (torch path) respectively -- same meaning
            either way: where the loaded tensors should live.
        mmap: forwarded to `fast_torch_load` only (safetensors is
            always effectively mmap'd -- see its own docstring).
            Ignored, not an error, when dispatching to the safetensors
            path, since there's no non-mmap mode to choose between
            there.
        **kwargs: forwarded to whichever loader is chosen. Passing a
            kwarg the chosen loader doesn't accept raises the same
            `TypeError` it would if you'd called that loader directly
            -- this dispatcher doesn't silently swallow bad arguments.

    Returns whatever the chosen loader returns (see its docstring for
    the exact shape).
    """
    suffix = Path(src).suffix.lower()
    if suffix == ".safetensors":
        return fast_safetensors_load(src, device=device, **kwargs)
    return fast_torch_load(src, device=device, mmap=mmap, **kwargs)
