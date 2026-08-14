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
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ThreadPlan:
    """Result of a thread-count recommendation."""

    total_logical: int
    reserved: int
    recommended: int

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
) -> ThreadPlan:
    """Compute a `ThreadPlan` and apply it to the current process.

    This sets, best-effort:
      - `torch.set_num_threads(...)` if torch is importable.
      - `OMP_NUM_THREADS` / `MKL_NUM_THREADS` env vars (if `set_env`),
        which affect NumPy, MKL, and other OpenMP-based libraries —
        but only if the caller sets them *before* those libraries have
        already read the env at import time. Setting env vars after
        NumPy/MKL are already imported has no effect on this process;
        this is a Python-level limitation, not a bug here.

    Returns the `ThreadPlan` that was applied, so the caller can log or
    inspect it.
    """
    plan = recommended_threads(total=total, reserve=reserve, threads=threads)

    if set_env:
        os.environ.setdefault("OMP_NUM_THREADS", str(plan.recommended))
        os.environ.setdefault("MKL_NUM_THREADS", str(plan.recommended))

    try:
        import torch  # local import: keep this module importable without torch

        torch.set_num_threads(plan.recommended)
    except Exception:
        pass

    return plan
