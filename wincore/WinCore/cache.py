"""
Disk-backed sample cache — for datasets where preprocessing (decode,
resize, augment-once, tokenize) is more expensive than a raw read, this
caches the *processed* result on disk (ideally an SSD) so epoch 2+
skip redoing that work, with an LRU byte budget and Windows-safe
writes (reuses `WinCore.io.atomic_write` so a half-written cache entry
never gets read as valid).

Why this exists
----------------
This is not a new caching algorithm and not a claim to beat
`functools.lru_cache` or a real KV store for what those tools are for.
It solves one specific, common gap: PyTorch's `Dataset.__getitem__` has
no built-in disk cache, so people either (a) recompute preprocessing
every single epoch even though the input didn't change, or (b) write
an ad-hoc pickle-to-disk cache with no eviction that eventually fills
the drive. This gives a small, dependency-free version of that with:

  - **LRU eviction with a byte budget**, not "cache forever" -- so it's
    safe to point at a large dataset without manually managing disk
    space.
  - **Content-addressed by a key you provide** (usually the dataset
    index, or a hash of the raw sample if it can change) -- this module
    does not invent a hashing scheme for your data; you decide what
    identifies a cache entry.
  - **Atomic writes** (via `WinCore.io.atomic_write`) so a process
    killed mid-write, or two DataLoader workers racing to write the
    same key, never leaves a corrupt/partial cache file that a later
    read would silently deserialize as valid.

What this deliberately is NOT
------------------------------
  - Not a distributed cache -- this is a single-machine, single local
    directory. If you need a shared cache across machines, that's a
    different tool (e.g. a network filesystem or a real KV store).
  - Not transparent/automatic -- you call `.get_or_compute(key, fn)`
    explicitly in `__getitem__`; nothing here monkeypatches `Dataset`.
  - Not a guarantee your drive is actually an SSD -- this module has no
    portable way to query storage media type from Python, so "SSD
    cache" describes the intended use (fast random-access storage),
    not something this module verifies or enforces.
"""
from __future__ import annotations

import hashlib
import os
import pickle
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Optional, Union

from .io import atomic_write


_FALLBACK_MAX_BYTES = 10 * 1024**3  # used only if disk_usage() itself fails (unusual)


def _auto_size_from_free_space(directory: Path, free_space_fraction: float) -> int:
    """`free_space_fraction` of the CURRENT free space on `directory`'s
    drive, via `shutil.disk_usage` -- a real measurement of this
    machine, not a guessed constant. Falls back to
    `_FALLBACK_MAX_BYTES` (with no crash) only if `disk_usage` itself
    raises, which is unusual but not impossible (e.g. a network drive
    that doesn't support the underlying syscall)."""
    import shutil

    try:
        free = shutil.disk_usage(directory).free
    except OSError:
        return _FALLBACK_MAX_BYTES
    return int(free * free_space_fraction)


def _key_to_filename(key: Any) -> str:
    """Turn an arbitrary hashable key into a filesystem-safe filename.
    Uses a stable hash rather than `str(key)` directly, since arbitrary
    keys (tuples, long strings, non-ASCII) aren't all valid on every
    filesystem -- this sidesteps that instead of trying to sanitize
    every possible input."""
    raw = repr(key).encode("utf-8", "surrogatepass")
    return hashlib.sha1(raw).hexdigest() + ".pkl"


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    bytes_on_disk: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total) if total else 0.0


class DiskCache:
    """LRU disk cache for expensive-to-compute, cheap-to-store sample
    data (preprocessed tensors, decoded images, tokenized text).

    `max_bytes`, if you don't set it explicitly, is NOT a fixed
    hardcoded number -- it's computed from the ACTUAL free space on
    `directory`'s drive at construction time (`free_space_fraction` of
    it, default 50%), via `shutil.disk_usage`. A fixed default (the
    previous behavior: always 10GB) is wrong in both directions across
    different machines -- overly conservative on a workstation with a
    2TB free NVMe drive, and a real problem on a laptop with 20GB free
    where 10GB is most of the remaining disk. Auto-sizing to what's
    actually there, once, at cache-creation time, fits the machine
    instead of guessing. It is NOT re-measured afterward -- if the
    drive fills up from something else later, this cache still respects
    the budget it originally computed, it does not shrink to make room.

    Always pass an explicit `max_bytes` yourself if you want a specific,
    predictable number instead (e.g. to leave headroom for other things
    that will also write to this drive during the run) -- an explicit
    value always wins over the auto-sizing.

    Example:
        cache = WinCore.cache.DiskCache("D:/wincore_cache", max_bytes=20 * 1024**3)

        # or let it size itself to the drive's free space:
        cache = WinCore.cache.DiskCache("D:/wincore_cache")

        class MyDataset(torch.utils.data.Dataset):
            def __getitem__(self, idx):
                return cache.get_or_compute(idx, lambda: self._load_and_preprocess(idx))
    """

    def __init__(
        self,
        directory: Union[str, Path],
        max_bytes: Optional[int] = None,
        free_space_fraction: float = 0.5,
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if max_bytes is None:
            max_bytes = _auto_size_from_free_space(self.directory, free_space_fraction)
        self.max_bytes = max_bytes
        self.stats = CacheStats()
        self._lock = RLock()
        # OrderedDict used purely as an in-process LRU index (filename ->
        # size, last-access order) -- the source of truth for *presence*
        # is always the actual file on disk, so a cache dir shared across
        # process restarts is still read correctly; this index just
        # avoids re-`stat`-ing every file to decide eviction order.
        self._index: "OrderedDict[str, int]" = OrderedDict()
        self._load_existing_index()

    def _load_existing_index(self) -> None:
        total = 0
        entries = []
        for p in self.directory.glob("*.pkl"):
            try:
                st = p.stat()
                entries.append((p.name, st.st_size, st.st_mtime))
                total += st.st_size
            except OSError:
                continue
        # Oldest mtime first, so the initial LRU order approximates
        # actual prior access order (best-effort -- mtime is a proxy
        # for last-write, not last-read, if the OS/filesystem doesn't
        # track atime).
        for name, size, _mtime in sorted(entries, key=lambda e: e[2]):
            self._index[name] = size
        self.stats.bytes_on_disk = total

    def get_or_compute(self, key: Any, compute_fn: Callable[[], Any]) -> Any:
        """Return the cached value for `key` if present, else call
        `compute_fn()`, store the result, and return it. `compute_fn`
        takes no arguments by design -- close over whatever it needs
        (e.g. `lambda: self._load(idx)`), which keeps this cache
        agnostic to what a "key" means for your dataset."""
        filename = _key_to_filename(key)
        path = self.directory / filename

        with self._lock:
            if filename in self._index and path.exists():
                self._index.move_to_end(filename)
                self.stats.hits += 1
                cache_hit = True
            else:
                cache_hit = False

        if cache_hit:
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except (OSError, pickle.PickleError):
                # Corrupt/partial entry despite atomic_write (e.g. disk
                # error after the fact) -- fall through and recompute
                # rather than raising out of a DataLoader worker.
                with self._lock:
                    self._index.pop(filename, None)

        with self._lock:
            self.stats.misses += 1

        value = compute_fn()
        self._store(filename, path, value)
        return value

    def _store(self, filename: str, path: Path, value: Any) -> None:
        def _write(tmp_path: Path) -> None:
            with open(tmp_path, "wb") as f:
                pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)

        atomic_write(_write, path)
        size = path.stat().st_size

        with self._lock:
            self._index[filename] = size
            self._index.move_to_end(filename)
            self.stats.bytes_on_disk += size
            self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        # Caller must hold self._lock.
        while self.stats.bytes_on_disk > self.max_bytes and self._index:
            oldest_name, oldest_size = next(iter(self._index.items()))
            self._index.popitem(last=False)
            try:
                (self.directory / oldest_name).unlink(missing_ok=True)
            except OSError:
                pass
            self.stats.bytes_on_disk -= oldest_size
            self.stats.evictions += 1

    def clear(self) -> None:
        """Remove every cached entry and reset stats-on-disk tracking
        (hit/miss counters are left alone -- those describe this
        process's history, not what's currently stored)."""
        with self._lock:
            for name in list(self._index.keys()):
                try:
                    (self.directory / name).unlink(missing_ok=True)
                except OSError:
                    pass
            self._index.clear()
            self.stats.bytes_on_disk = 0

    def __len__(self) -> int:
        return len(self._index)
