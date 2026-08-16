"""
CPU thread/core auto-scheduler.

Why this exists
----------------
Letting a training process claim every logical thread on the machine
usually backfires: the OS scheduler, background services, and (on
Windows especially) antivirus/indexer activity all compete for the same
cores, so 100%-of-threads settings often run *slower* and less stably
than leaving a small reserve free for the system.

There is no universally "correct" number of threads to reserve — it
depends on how many the machine has in the first place. This module
encodes a simple, tiered heuristic:

    total logical threads   reserved for the OS   used by default
    ----------------------  ---------------------  ---------------
    <= 4                    1                       total - 1
    5-8                     1                       total - 1
    9-12                    2                       total - 2
    13-16                   2-3                     total - 2/3
    > 16                    up to 4 (scales down)    total - reserve

This is a *default*, not a hard rule — pass `reserve=` or `threads=`
explicitly to override it. Nothing here inspects CPU model/generation;
it only counts logical threads, which is the one thing Python can read
reliably and portably.

What `apply()` actually touches (read this before assuming it does more)
--------------------------------------------------------------------
`torch.set_num_threads()` / `OMP_NUM_THREADS` / `MKL_NUM_THREADS` only
size the thread pool *inside* a single torch/BLAS op (matmul, conv,
elementwise kernels, ...). They do not, and structurally cannot, change
how many OS threads a plain Python `for` loop uses, because a Python
loop doesn't consult those thread pools at all — that is a GIL-bound,
single-thread-of-Python-bytecode question, not a BLAS-op question. No
amount of retuning those two settings closes that gap; it is a
different layer of the stack. That's why this module also offers real
OS-level scheduling controls below (`set_priority`, `pin_affinity`),
which act on the *process*, not on a library's internal thread pool —
those do affect Python loops, DataLoader worker processes, and every
other thread in the process, because the OS scheduler is what's being
configured, not torch/BLAS.
"""
from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass
from typing import Optional


class PriorityError(RuntimeError):
    """Raised when a priority/affinity change was requested but could
    not be applied (missing psutil, insufficient OS permissions, or an
    unsupported platform). `apply(..., strict=False)` (the default)
    catches this internally and just reports the failure in
    `AppliedPlan.warnings` instead of raising, since a training run
    should not crash because it couldn't renice itself."""


# Cross-platform priority tiers, expressed as psutil's own priority
# constants so the mapping is exact instead of a mood-based synonym
# table. On POSIX these read from `nice`-equivalent priority classes
# psutil emulates; on Windows they're the real Win32 priority classes
# (IDLE_PRIORITY_CLASS ... HIGH_PRIORITY_CLASS) via SetPriorityClass.
_PRIORITY_LEVELS = ("idle", "below_normal", "normal", "above_normal", "high")


def _psutil_priority_value(level: str):
    import psutil  # local import: optional dependency (the `sysinfo` extra)

    if platform.system() == "Windows":
        table = {
            "idle": psutil.IDLE_PRIORITY_CLASS,
            "below_normal": psutil.BELOW_NORMAL_PRIORITY_CLASS,
            "normal": psutil.NORMAL_PRIORITY_CLASS,
            "above_normal": psutil.ABOVE_NORMAL_PRIORITY_CLASS,
            "high": psutil.HIGH_PRIORITY_CLASS,
        }
    else:
        # POSIX `nice` range is -20 (highest) .. 19 (lowest). These are
        # deliberately mild (no negative/real-time values) -- a training
        # script asking for real-time priority on someone's desktop is
        # how you make their mouse cursor stop moving, and doing so
        # usually requires root anyway. "high" here is a nice, cooperative
        # nudge above default, not a real-time-scheduling request.
        table = {
            "idle": 15,
            "below_normal": 5,
            "normal": 0,
            "above_normal": -5,
            "high": -10,
        }
    return table[level]


def set_priority(level: str = "above_normal", pid: Optional[int] = None) -> str:
    """Set the OS scheduling priority of this (or another) process.

    This is a real OS-level call -- `SetPriorityClass` on Windows,
    `setpriority`/`nice` on POSIX, via `psutil.Process().nice(...)` --
    not a torch/BLAS setting. It changes how the OS scheduler treats
    *every* thread in the process (Python main loop, DataLoader worker
    processes if you pass their pids, everything), which is the piece
    `torch.set_num_threads()` cannot reach.

    Args:
        level: one of "idle", "below_normal", "normal", "above_normal",
            "high". Default "above_normal" -- enough to reduce
            preemption by background services without starving the
            rest of the system the way "high"/real-time would.
        pid: process to adjust; defaults to the current process. Pass
            a DataLoader worker's pid (e.g. from a `worker_init_fn`) to
            raise its priority too, since worker processes are not
            covered by adjusting only the main process.

    Returns the level string that was applied.

    Raises `PriorityError` if psutil isn't installed, the level name is
    invalid, or the OS denies the change (e.g. lowering another
    process's niceness without the right privileges on POSIX). Lowering
    your *own* process's niceness, or any change on Windows via
    SetPriorityClass, does not require elevated privileges.
    """
    if level not in _PRIORITY_LEVELS:
        raise PriorityError(f"Unknown priority level '{level}'. Choose from: {_PRIORITY_LEVELS}.")
    try:
        import psutil
    except ImportError as exc:
        raise PriorityError(
            "set_priority() needs psutil (pip install WinCore[sysinfo], "
            "or plain `pip install psutil`)."
        ) from exc

    try:
        proc = psutil.Process(pid) if pid is not None else psutil.Process()
        proc.nice(_psutil_priority_value(level))
    except psutil.Error as exc:
        raise PriorityError(f"OS refused priority change to '{level}': {exc}") from exc
    return level


def _detect_windows_performance_cores() -> Optional[list]:
    """Best-effort: return the logical CPU indices Windows itself
    classifies as performance cores (P-cores), or `None` if that can't
    be determined (not Windows, an OS build too old to report it, or
    any ctypes failure -- this must never raise, only degrade to
    "don't know").

    Why this exists: `range(recommended_threads().recommended)` -- the
    previous, and still the fallback, default for `cpus` -- just picks
    the first N logical CPU *indices*. That's an assumption, not a
    measurement: nothing about "which logical index number Windows
    assigns" guarantees those are the fast cores on a P-core/E-core
    hybrid CPU (Intel 12th gen+). On several real Alder Lake/Raptor
    Lake layouts the low indices are NOT all P-cores. Pinning "the
    first N indices" on such a machine can silently pin the process
    onto E-cores instead -- the opposite of what affinity pinning is
    for, per this module's own docstring above.

    psutil has no cross-platform concept of core type, so this can't
    be done through it. It's read here directly via
    `GetLogicalProcessorInformationEx(RelationProcessorCore, ...)`
    (Windows 10 20348+ / Windows 11), which reports one entry per
    physical core with an `EfficiencyClass` byte -- higher means more
    performant, per Microsoft's own docs. This picks the logical CPUs
    belonging to cores at the highest EfficiencyClass seen.
    """
    if platform.system() != "Windows":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32
        RelationProcessorCore = 0

        length = wintypes.DWORD(0)
        kernel32.GetLogicalProcessorInformationEx(RelationProcessorCore, None, ctypes.byref(length))
        if length.value == 0:
            return None  # call itself unsupported on this build

        buf = ctypes.create_string_buffer(length.value)
        ok = kernel32.GetLogicalProcessorInformationEx(
            RelationProcessorCore, buf, ctypes.byref(length)
        )
        if not ok:
            return None

        # Walk the variable-length SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX
        # records by hand (ctypes has no built-in support for this
        # union-of-variable-length-arrays Win32 layout): each record
        # starts with (Relationship: DWORD, Size: DWORD), then for
        # RelationProcessorCore a PROCESSOR_RELATIONSHIP: Flags (BYTE),
        # EfficiencyClass (BYTE), Reserved[20], GroupCount (WORD),
        # then GroupCount x GROUP_AFFINITY{ Mask: ULONG_PTR, Group: WORD,
        # Reserved[3]: WORD }.
        cores = []  # list of (efficiency_class, [logical cpu indices])
        offset = 0
        ptr_size = ctypes.sizeof(ctypes.c_void_p)
        while offset < length.value:
            relationship = int.from_bytes(buf.raw[offset : offset + 4], "little")
            size = int.from_bytes(buf.raw[offset + 4 : offset + 8], "little")
            if size == 0:
                break
            if relationship == RelationProcessorCore:
                p = offset + 8
                efficiency_class = buf.raw[p + 1]
                group_count = int.from_bytes(buf.raw[p + 22 : p + 24], "little")
                gbase = p + 24
                cpus_for_core = []
                for g in range(group_count):
                    gp = gbase + g * (ptr_size + 8)
                    mask = int.from_bytes(buf.raw[gp : gp + ptr_size], "little")
                    bit = 0
                    while mask:
                        if mask & 1:
                            cpus_for_core.append(bit)
                        mask >>= 1
                        bit += 1
                cores.append((efficiency_class, cpus_for_core))
            offset += size

        if not cores:
            return None

        best_class = max(c for c, _ in cores)
        performance_cpus = sorted(
            {cpu for eff, cpus in cores if eff == best_class for cpu in cpus}
        )
        # A uniform (non-hybrid) CPU reports one EfficiencyClass for
        # every core -- in that case "performance cores" is just "all
        # cores", which is a no-op, not wrong, so still return it.
        return performance_cpus or None
    except Exception:
        # Any ctypes/struct-layout surprise (build mismatch, etc.) --
        # degrade to "couldn't determine it", never raise from here.
        return None


def _select_pin_cpus(recommended: int) -> list:
    """Single shared decision for "which `recommended` logical CPU
    indices should a pin target" -- used by BOTH `pin_affinity()`'s
    own default (`cpus=None`) AND by `apply()` when it pins on the
    caller's behalf. This function existing at all is itself the fix
    for a real integration bug: `apply(affinity=True)` -- the path
    documented in the Quick Start, and what almost every real caller
    actually uses instead of calling `pin_affinity()` directly --
    used to build `list(range(plan.recommended))` itself and pass
    that in as an explicit `cpus` argument. Passing an explicit `cpus`
    bypasses `pin_affinity()`'s `if cpus is None:` branch entirely, so
    the P-core detection in `_detect_windows_performance_cores()` only
    ever ran for someone calling `pin_affinity()` with no arguments --
    `apply()` itself, the common path, silently overrode it back to
    plain `range(N)` on every call. A per-module unit test of
    `pin_affinity()` alone can't catch this class of bug because
    `pin_affinity()` in isolation behaves exactly as documented; only
    an integration-level test that goes through `apply()` and asserts
    on the *actual* cpus passed to `psutil` would surface it (added
    below: `test_apply_affinity_uses_p_core_detection_not_bare_range`).

    `recommended` is the caller's own already-decided thread count
    (`plan.recommended` inside `apply()`, which can differ from
    `recommended_threads().recommended` if the caller passed
    `total`/`reserve`/`threads` overrides) -- this function only
    decides WHICH `recommended` cpu indices to use, never how many.
    """
    p_cores = _detect_windows_performance_cores()
    if p_cores:
        # Use up to `recommended` of the detected P-core logical CPUs.
        # If there are fewer P-core threads than `recommended` (e.g. a
        # low-P-core-count hybrid part), use all of them rather than
        # spilling onto E-cores -- under-using thread count is a
        # smaller cost than scheduling training compute onto
        # efficiency cores.
        return p_cores[:recommended] if len(p_cores) >= recommended else p_cores
    return list(range(recommended))


def pin_affinity(cpus: Optional[list] = None, pid: Optional[int] = None) -> list:
    """Pin this (or another) process to a specific set of logical CPUs.

    Also a real OS-level call (`SetProcessAffinityMask` on Windows,
    `sched_setaffinity` on Linux, via `psutil.Process().cpu_affinity()`)
    -- it restricts which cores the OS scheduler is allowed to run this
    process's threads on at all, which is a different lever from
    thread-pool *sizing*: sizing controls how many threads a BLAS op
    spins up; affinity controls which physical cores any of the
    process's threads (including plain Python ones) are scheduled onto.
    Pinning reduces cross-core cache-line bouncing and OS migration
    jitter, which matters most on machines with mixed core types
    (Intel P-core/E-core hybrid CPUs) where an unpinned process can get
    bounced onto slower efficiency cores mid-run.

    Args:
        cpus: list of logical CPU indices to restrict to. If not
            given, uses `_select_pin_cpus()` against
            `recommended_threads().recommended` -- on Windows, this
            first tries to detect actual performance-core (P-core)
            logical indices via `GetLogicalProcessorInformationEx` and
            pin to exactly those; otherwise (and as the fallback)
            falls back to `range(recommended_threads().recommended)`.
            On a hybrid CPU, blindly using `range(N)` risks pinning to
            logical indices that Windows happens to assign to
            E-cores, which is the opposite of what pinning is for
            here -- see `_detect_windows_performance_cores()`. Note
            that `apply(affinity=True)` does NOT go through this
            default -- it calls `_select_pin_cpus()` itself against
            its own `plan.recommended`, so passing an explicit `cpus`
            here and calling through `apply()` both get the same
            P-core-aware selection.
        pid: defaults to the current process.

    Returns the list of CPU indices actually applied.

    Raises `PriorityError` if psutil isn't installed or the platform
    doesn't support CPU affinity (notably macOS -- the OS provides no
    public API for hard affinity pinning there; this is a platform
    limitation, not something WinCore can work around).
    """
    try:
        import psutil
    except ImportError as exc:
        raise PriorityError(
            "pin_affinity() needs psutil (pip install WinCore[sysinfo], "
            "or plain `pip install psutil`)."
        ) from exc

    if cpus is None:
        cpus = _select_pin_cpus(recommended_threads().recommended)

    try:
        proc = psutil.Process(pid) if pid is not None else psutil.Process()
        proc.cpu_affinity(cpus)
        applied = proc.cpu_affinity()
    except (AttributeError, NotImplementedError) as exc:
        raise PriorityError(
            "cpu_affinity() isn't supported on this platform (e.g. macOS "
            "provides no public API for it)."
        ) from exc
    except psutil.Error as exc:
        raise PriorityError(f"OS refused affinity change to {cpus}: {exc}") from exc
    return applied


@dataclass(frozen=True)
class ThreadPlan:
    """Result of a thread-count recommendation."""

    total_logical: int
    reserved: int
    recommended: int
    warnings: tuple = ()

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"ThreadPlan(total_logical={self.total_logical}, "
            f"reserved={self.reserved}, recommended={self.recommended})"
        )


def _default_reserve(total: int) -> int:
    if total <= 8:
        return 1 if total > 1 else 0
    if total <= 12:
        return 2
    if total <= 16:
        return 3
    # Scales down gradually for very high thread counts, capped at 4
    # so large workstation/server CPUs don't reserve an excessive slice.
    return min(4, max(2, total // 8))


def recommended_threads(
    total: Optional[int] = None,
    reserve: Optional[int] = None,
    threads: Optional[int] = None,
) -> ThreadPlan:
    """Compute how many logical threads a process should use.

    Args:
        total: override the detected logical thread count (mainly for
            testing). Defaults to `os.cpu_count()`.
        reserve: force this many threads to be left free for the OS,
            instead of the tiered default.
        threads: skip the heuristic entirely and use exactly this many
            threads (still clamped to `[1, total]`).

    Returns:
        A `ThreadPlan` with the detected total, the number reserved,
        and the recommended thread count to actually use.
    """
    detected_total = os.cpu_count() or 1
    total = detected_total if total is None else total
    total = max(1, total)

    if threads is not None:
        recommended = max(1, min(threads, total))
        reserved = total - recommended
        return ThreadPlan(total_logical=total, reserved=reserved, recommended=recommended)

    reserved = _default_reserve(total) if reserve is None else max(0, reserve)
    recommended = max(1, total - reserved)
    return ThreadPlan(total_logical=total, reserved=reserved, recommended=recommended)


def apply(
    total: Optional[int] = None,
    reserve: Optional[int] = None,
    threads: Optional[int] = None,
    set_env: bool = True,
    priority: Optional[str] = None,
    affinity: bool = False,
    strict: bool = False,
) -> ThreadPlan:
    """Compute a `ThreadPlan` and apply it to the current process.

    This always sets, best-effort:
      - `torch.set_num_threads(...)` if torch is importable.
      - `OMP_NUM_THREADS` / `MKL_NUM_THREADS` env vars (if `set_env`),
        which affect NumPy, MKL, and other OpenMP-based libraries —
        but only if the caller sets them *before* those libraries have
        already read the env at import time. Setting env vars after
        NumPy/MKL are already imported has no effect on this process;
        this is a Python-level limitation, not a bug here.

    Optionally, when the caller asks for them, it also applies the
    OS-level controls from this module that thread-pool sizing cannot
    reach (see module docstring):
      - `priority`: one of the levels accepted by `set_priority()`
        (e.g. "above_normal"). `None` (default) leaves OS priority
        untouched.
      - `affinity`: if True, calls `pin_affinity()` with the same
        P-core-aware CPU selection `pin_affinity()` uses by default
        (`_select_pin_cpus()`, sized to `plan.recommended` -- this
        call's own thread count, which may differ from
        `recommended_threads().recommended` if `total`/`reserve`/
        `threads` were overridden above), NOT a bare
        `range(plan.recommended)`. A prior version of this function
        built that bare range itself and passed it to `pin_affinity()`
        as an explicit `cpus` argument -- which bypassed
        `pin_affinity()`'s own P-core detection entirely (that
        detection only runs when `cpus is None`), silently undoing it
        for every caller going through `apply()`, i.e. almost everyone,
        since `apply()` is the Quick Start entry point and
        `pin_affinity()` is rarely called directly.

    Both of the above need `psutil` and can fail for OS/permission
    reasons that have nothing to do with thread counting. By default
    (`strict=False`) a failure there is recorded in
    `plan.warnings` and does not raise or block training from starting
    — a run should not crash just because it couldn't renice itself.
    Pass `strict=True` to raise `PriorityError` instead when you want
    to be certain the OS-level request actually took effect (e.g. in a
    benchmarking harness where an unpinned run would invalidate the
    result).

    Returns the `ThreadPlan` that was applied, with `.warnings`
    populated with a list of any non-fatal issues encountered.
    """
    plan = recommended_threads(total=total, reserve=reserve, threads=threads)
    warnings: list = []

    if set_env:
        os.environ.setdefault("OMP_NUM_THREADS", str(plan.recommended))
        os.environ.setdefault("MKL_NUM_THREADS", str(plan.recommended))

    try:
        import torch  # local import: keep this module importable without torch

        torch.set_num_threads(plan.recommended)
    except Exception:
        pass

    if priority is not None:
        try:
            set_priority(priority)
        except PriorityError as exc:
            if strict:
                raise
            warnings.append(str(exc))

    if affinity:
        try:
            pin_affinity(_select_pin_cpus(plan.recommended))
        except PriorityError as exc:
            if strict:
                raise
            warnings.append(str(exc))

    if warnings:
        plan = ThreadPlan(
            total_logical=plan.total_logical,
            reserved=plan.reserved,
            recommended=plan.recommended,
            warnings=tuple(warnings),
        )
    return plan
