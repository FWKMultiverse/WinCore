"""
Memory manager — DataLoader defaults, VRAM-pressure-aware cache
clearing, and pinned-memory transfer helpers.

Why this exists
----------------
None of this reimplements PyTorch's CUDA caching allocator or the
DataLoader's own multiprocessing -- it configures both more carefully
than the library defaults, and specifically accounts for two things
Windows does worse than Linux:

  - **`torch.multiprocessing` workers are heavier to spawn on Windows.**
    Windows has no `fork()`; every DataLoader worker is spawned fresh
    (`spawn` start method), which is slower per-worker to start and
    re-imports your script's module-level code in each worker. This
    means the "just crank up num_workers" advice that works on Linux
    can backfire on Windows -- too many workers spend more time
    starting up / re-importing than they save. `recommended_dataloader_kwargs()`
    accounts for this with a lower default ceiling on Windows.
  - **`torch.cuda.empty_cache()` calling it every step is a common
    anti-pattern** -- it doesn't reduce peak usage (PyTorch's caching
    allocator already reuses freed blocks internally) and forces a
    CUDA sync, which stalls the pipeline. `CacheGuard` only calls it
    when free VRAM has actually dropped below a threshold, not on a
    fixed schedule.

What this deliberately is NOT
------------------------------
  - Not a replacement for PyTorch's caching allocator -- `CacheGuard`
    calls the same `torch.cuda.empty_cache()` you'd call yourself, just
    conditionally instead of unconditionally.
  - Not a leak detector -- if VRAM usage climbs step over step and
    `empty_cache()` doesn't reclaim it, that is a real reference (e.g. a
    growing Python list holding onto tensors), and this module will
    keep reporting pressure rather than hide it.
"""
from __future__ import annotations

import platform
from collections import deque
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DataLoaderPlan:
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: Optional[int]
    reason: str


def recommended_dataloader_kwargs(
    cpu_recommended_threads: Optional[int] = None,
    cuda_available: Optional[bool] = None,
) -> DataLoaderPlan:
    """Recommend `DataLoader(**kwargs)` values for this machine.

    Args:
        cpu_recommended_threads: use `WinCore.cpu.recommended_threads().recommended`
            if you already computed it, to avoid detecting CPU count
            twice. Detected fresh via `os.cpu_count()` if omitted.
        cuda_available: pass `torch.cuda.is_available()` if you already
            know it; detected lazily (torch import, best-effort) if
            omitted, and treated as False if torch import fails.
    """
    import os

    if cpu_recommended_threads is None:
        from .cpu import recommended_threads

        cpu_recommended_threads = recommended_threads().recommended

    if cuda_available is None:
        try:
            import torch

            cuda_available = torch.cuda.is_available()
        except Exception:
            cuda_available = False

    is_windows = platform.system() == "Windows"

    # Windows spawns each worker fresh (no fork()) and re-imports your
    # script's module-level code per worker -- more workers than this
    # tends to spend more wall-clock starting up than it saves reading
    # ahead. Linux forks, so it can use closer to the full thread budget.
    if is_windows:
        num_workers = max(0, min(cpu_recommended_threads, 6))
        reason = (
            "Windows DataLoader workers use the 'spawn' start method "
            "(no fork()), which re-imports your script per worker -- "
            "capped lower than the raw CPU count so worker startup "
            "overhead doesn't outweigh the parallel read-ahead."
        )
    else:
        num_workers = max(0, cpu_recommended_threads)
        reason = "fork() on Linux/Mac makes worker startup cheap, so this uses closer to the full recommended thread budget."

    return DataLoaderPlan(
        num_workers=num_workers,
        pin_memory=bool(cuda_available),
        persistent_workers=num_workers > 0,  # avoids re-spawning (expensive on Windows) every epoch
        prefetch_factor=4 if num_workers > 0 else None,
        reason=reason,
    )


def to_device_non_blocking(tensor, device, pin_memory_used: bool = True):
    """`tensor.to(device, non_blocking=True)` only actually overlaps
    with compute if the source tensor is in pinned host memory -- doing
    this with a non-pinned tensor silently behaves like a blocking copy
    (no error, just no overlap, which is easy to not notice). This
    wrapper is a documented, one-line reminder wired to the actual
    pinned-memory precondition rather than a copy-pasted `non_blocking=True`
    that may or may not be doing anything on a given tensor."""
    return tensor.to(device, non_blocking=pin_memory_used)


class WorkingSetTrimError(RuntimeError):
    """Raised by `trim_working_set()` when the OS call itself fails
    (missing privileges, or a non-Windows platform without a real
    equivalent). Not raised for "nothing to trim" -- that's a normal,
    silent no-op."""


def trim_working_set() -> bool:
    """Ask Windows to release freed-but-still-resident pages from this
    process's *working set* back to the OS, so Task Manager / Process
    Explorer's reported memory actually drops after a large allocation
    (e.g. a big batch, a dataset loaded fully into RAM) is freed.

    Why this is needed at all
    --------------------------
    Freeing Python objects (`del big_list`, a batch going out of scope,
    `gc.collect()`) only returns that memory to the *process's own*
    allocator (CPython's pymalloc / the C heap under it) -- it does NOT
    by itself tell the OS "you can reclaim these physical pages". On
    Windows in particular, the process's working set (the physical RAM
    pages Windows has actually mapped in for it) tends to stay high
    after a spike even once the allocator has internally freed the
    memory, because Windows has no proactive equivalent to Linux's
    `malloc_trim()` / the kernel's more aggressive page reclaim under
    memory pressure -- it waits for *external* memory pressure before
    trimming a process's working set on its own. The practical symptom
    reported by users: Task Manager shows the training process sitting
    at, say, 20GB RAM long after the one large batch/preprocessing step
    that actually needed that much has finished and been garbage
    collected -- the memory is genuinely free from Python's point of
    view, the OS just hasn't been told to reclaim the physical pages.

    This calls the real Win32 API for that: `SetProcessWorkingSetSize`
    with `(-1, -1)`, a documented special case meaning "trim the
    working set to the minimum now" (see Microsoft's docs for
    `SetProcessWorkingSetSizeEx` / the legacy `SetProcessWorkingSetSize`
    signature) -- via `ctypes`, no extra dependency.

    Important caveats (read before wiring this into a hot loop)
    -------------------------------------------------------------
      - This does NOT reduce your actual peak memory *usage* -- it only
        returns already-freed pages to the OS sooner than Windows would
        on its own. Calling it does not prevent an OOM if the memory is
        still genuinely in use.
      - Trimmed pages will simply be paged back in (a page fault, which
        has a real cost) the next time they're touched -- if you call
        this every step in a loop that immediately reallocates similar-
        sized buffers, you can make things *slower*, not faster, by
        forcing repeated trim-then-refault cycles. Call it at natural
        low-memory points (after a preprocessing pass finishes, between
        epochs, after `del`-ing something genuinely large), not on a
        tight per-step schedule.
      - It is a real OS call, not a Python-level trick -- it changes
        what Task Manager reports, it does not change what
        `torch.cuda.memory_allocated()` or Python's own memory
        accounting report (those were already correct).

    Returns:
        `True` if the trim call was issued (Windows only).
        `False` on any other platform -- a deliberate, silent no-op,
        since there's nothing analogous to call safely across every
        POSIX system from pure Python without extra dependencies (glibc
        `malloc_trim()` exists but isn't portable to musl/BSD/macOS).

    Raises `WorkingSetTrimError` only if this IS Windows and the actual
    Win32 call reports failure (rare -- typically means something else
    is wrong with the process, e.g. it's already being torn down).
    """
    if platform.system() != "Windows":
        return False

    import ctypes

    handle = ctypes.windll.kernel32.GetCurrentProcess()
    # SetProcessWorkingSetSize(handle, -1, -1) is the documented sentinel
    # for "trim to minimum now" rather than "set an exact size".
    ok = ctypes.windll.kernel32.SetProcessWorkingSetSize(handle, -1, -1)
    if not ok:
        raise WorkingSetTrimError(
            f"SetProcessWorkingSetSize failed (GetLastError={ctypes.get_last_error()})"
        )
    return True


def estimate_worker_ram_multiplier(num_workers: int) -> str:
    """One-line explanation of a Windows-specific RAM trap that has no
    single-number fix, so this returns guidance, not a false-precision
    byte count.

    On Linux, `fork()`-based DataLoader workers share the parent's
    memory pages copy-on-write -- a large in-memory `Dataset` (e.g. a
    list/array holding the whole dataset) is NOT duplicated per worker
    unless a worker actually writes to it. On Windows, workers are
    created via `spawn`, which starts a fresh interpreter per worker
    and re-imports/re-pickles what the worker needs -- a `Dataset`
    holding a large in-memory structure gets a full independent copy
    IN EACH WORKER PROCESS, not shared. Practical effect: total RAM
    use is roughly `(num_workers + 1) x that structure's size` on
    Windows, vs. close to `1x` on Linux for the same code -- the same
    training script can be fine on Linux and OOM on Windows purely from
    this, with no leak or bug in the model code at all.

    This function does not (and cannot, from here) know your dataset's
    actual size, so it returns a short, concrete explanation string
    rather than a fabricated number -- use it as a printed hint next to
    `recommended_dataloader_kwargs()`'s own `num_workers`.
    """
    if num_workers <= 0:
        return "num_workers=0 runs in the main process -- no per-worker RAM duplication."
    return (
        f"num_workers={num_workers} on Windows: each worker is a separate "
        f"spawned process, not a fork() -- any large in-memory structure "
        f"your Dataset holds (a big list/array/dict) gets copied fully "
        f"into all {num_workers} workers, roughly {num_workers + 1}x the "
        f"single-process RAM for that structure. If your Dataset reads "
        f"from disk per-item instead of holding everything in memory "
        f"(or memory-maps it, e.g. via numpy.memmap), this doesn't apply."
    )


@dataclass
class MemoryPressureEvent:
    free_gb: float
    total_gb: float
    free_fraction: float
    cleared: bool
    predictive: bool = False  # True if this clear happened BEFORE crossing
    # min_free_fraction, based on the trend -- see CacheGuard(adaptive=True)


def _predict_future_fraction(history: list, lookahead: int) -> float:
    """Pure function (no torch dependency): given recent free_fraction
    readings (oldest first), fit a simple linear trend and extrapolate
    `lookahead` checks into the future. Ordinary least-squares slope
    over the reading index -- deliberately simple (no smoothing/decay)
    so its behavior is easy to reason about and to unit-test without a
    GPU; see tests/test_memory.py."""
    n = len(history)
    if n == 0:
        return 1.0
    if n < 2:
        return history[-1]
    xs = range(n)
    mean_x = (n - 1) / 2.0
    mean_y = sum(history) / n
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, history))
    denominator = sum((x - mean_x) ** 2 for x in xs)
    slope = numerator / denominator if denominator else 0.0
    return history[-1] + slope * lookahead


def _decide_clear(
    free_fraction: float,
    min_free_fraction: float,
    adaptive: bool,
    history: list,
    lookahead_checks: int,
    min_history_for_prediction: int = 3,
):
    """Pure decision function factored out of `CacheGuard.check()` so
    the actual clear-or-not logic (including the predictive branch) has
    direct unit coverage without needing CUDA. Returns
    `(should_clear: bool, predictive: bool)`."""
    if free_fraction < min_free_fraction:
        return True, False
    if adaptive and len(history) >= min_history_for_prediction:
        predicted = _predict_future_fraction(history, lookahead_checks)
        if predicted < min_free_fraction:
            return True, True
    return False, False


class CacheGuard:
    """Calls `torch.cuda.empty_cache()` only when free VRAM has
    actually dropped below `min_free_fraction` -- or, with
    `adaptive=True`, when the recent trend predicts it's ABOUT to --
    instead of every N steps on a fixed schedule (which forces a CUDA
    sync for no benefit on steps where there was no pressure to
    relieve).

    Reactive mode (default, `adaptive=False`) is unchanged from before:
    clears only after `free_fraction` has already dropped below
    `min_free_fraction`. That's a real, working safety net, but it
    means the clear happens at the same moment allocation pressure is
    highest -- the worst time for a CUDA sync, and one step later than
    ideal if usage is climbing fast.

    `adaptive=True` additionally keeps a short rolling history of
    `free_fraction` readings and fits a simple linear trend across
    them (see `_predict_future_fraction`). If that trend predicts
    `free_fraction` will cross `min_free_fraction` within the next
    `lookahead_checks` calls to `.check()`, it clears now, before the
    threshold is actually crossed -- catching a fast, steady climb
    before it becomes a hard OOM instead of right as it happens. This
    is a simple linear extrapolation over your own recent readings, not
    a model of PyTorch's allocator internals -- a sudden one-step spike
    (e.g. a single unusually large batch) won't be predicted in
    advance, only a trend visible across several checks. A
    `MemoryPressureEvent.predictive=True` clear means this trend logic
    fired; `predictive=False` means it was the plain reactive threshold
    (same as `adaptive=False` always produces).

    Example:
        guard = WinCore.memory.CacheGuard(min_free_fraction=0.10)
        for step, batch in enumerate(loader):
            train_step(batch)
            if step % 50 == 0:
                guard.check()

        # or, letting it catch a climbing trend before the hard threshold:
        guard = WinCore.memory.CacheGuard(min_free_fraction=0.10, adaptive=True)
    """

    def __init__(
        self,
        min_free_fraction: float = 0.10,
        gpu_index: int = 0,
        adaptive: bool = False,
        trend_window: int = 6,
        lookahead_checks: int = 3,
    ):
        self.min_free_fraction = min_free_fraction
        self.gpu_index = gpu_index
        self.adaptive = adaptive
        self.lookahead_checks = lookahead_checks
        self.last_event: Optional[MemoryPressureEvent] = None
        self._history: "deque[float]" = deque(maxlen=trend_window)

    def check(self) -> Optional[MemoryPressureEvent]:
        """Returns a `MemoryPressureEvent` if it inspected VRAM (even
        if it didn't need to clear anything), or `None` if CUDA isn't
        available -- never raises, safe to call unconditionally in a
        training loop that might run CPU-only."""
        try:
            import torch

            if not torch.cuda.is_available():
                return None
            free_b, total_b = torch.cuda.mem_get_info(self.gpu_index)
        except Exception:
            return None

        total_gb = total_b / (1024**3)
        free_gb = free_b / (1024**3)
        free_fraction = free_gb / total_gb if total_gb > 0 else 1.0

        history_for_prediction = list(self._history)  # BEFORE appending this reading
        should_clear, predictive = _decide_clear(
            free_fraction,
            self.min_free_fraction,
            self.adaptive,
            history_for_prediction,
            self.lookahead_checks,
        )
        self._history.append(free_fraction)

        if should_clear:
            import torch

            torch.cuda.empty_cache()

        event = MemoryPressureEvent(
            free_gb=round(free_gb, 3),
            total_gb=round(total_gb, 3),
            free_fraction=round(free_fraction, 4),
            cleared=should_clear,
            predictive=predictive,
        )
        self.last_event = event
        return event
