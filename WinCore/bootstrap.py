"""
One-call setup: `optimize()` applies the real, already-independently-
working optimizations from `WinCore.cpu` and `WinCore.precision` in the
one order that actually matters, instead of a script having to
remember (1) which functions to call, (2) that `cpu.apply()` must run
before anything else imports torch to have any effect at all (see that
module's docstring, and CHANGELOG 0.8.2 / bug #8.1), and (3) that
`precision.cuda_perf_defaults()` needs an actual CUDA device to have
anything to check.

What this module deliberately is NOT
--------------------------------------
This is a thin *sequencing* layer, not a new optimization technique.
Every effect `optimize()` has already exists, already has its own
tests, and already has its own honest docstring explaining exactly
what it does and doesn't do (`WinCore.cpu.apply`,
`WinCore.precision.cuda_perf_defaults`) -- read those for the real
mechanism. This module's only job is calling them in the right order
and handing back one combined report instead of two separate ones you
have to remember to check.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class OptimizePlan:
    """Combined result of `optimize()`. `.cpu` is always a
    `WinCore.cpu.ThreadPlan` (CPU thread scheduling has no
    hardware-availability precondition the way CUDA tuning does).
    `.cuda` is a `WinCore.precision.CudaPerfPlan` if `apply_cuda=True`
    (default) — including the inert all-`False` plan
    `cuda_perf_defaults()` itself returns when there's no CUDA device,
    which is a normal, expected result on a CPU-only machine, not a
    failure — or `None` if `apply_cuda=False` was passed, meaning CUDA
    tuning wasn't even attempted."""

    cpu: "object"  # WinCore.cpu.ThreadPlan
    cuda: Optional["object"]  # WinCore.precision.CudaPerfPlan, or None if apply_cuda=False
    warnings: List[str] = field(default_factory=list)


def optimize(
    cpu_kwargs: Optional[dict] = None,
    cuda_kwargs: Optional[dict] = None,
    apply_cuda: bool = True,
) -> OptimizePlan:
    """Apply `WinCore.cpu.apply()` then (if a CUDA device is present)
    `WinCore.precision.cuda_perf_defaults()`, in that order, and return
    both results together.

    Call this as the FIRST WinCore-related thing your script does --
    ideally the first import-adjacent line at all, before your own
    code imports torch for any other reason. The ordering requirement
    is real and specific, not generic advice: `WinCore.cpu.apply()`
    sets `OMP_NUM_THREADS`/`MKL_NUM_THREADS` via
    `os.environ.setdefault(...)`, and those only affect torch/OMP/MKL
    thread-pool sizing if they're set *before* torch's own
    initialization reads them -- once torch has been imported anywhere
    in the process (by your code, by a library you imported, or by
    WinCore itself), setting the env var later has no effect. `import
    WinCore` alone does NOT import torch (see CHANGELOG 0.8.2, bug
    #8.1) specifically so that calling `WinCore.optimize()` right after
    `import WinCore` still has a chance to matter -- but if your script
    does `import torch` (directly or via another library) before
    calling this, the CPU thread-pool sizing part of this call is
    already too late for that specific effect, same as calling
    `WinCore.cpu.apply()` directly would be.

    Args:
        cpu_kwargs: forwarded as `**kwargs` to `WinCore.cpu.apply()`
            (e.g. `{"priority": "above_normal", "affinity": True}`).
            `None` (default) uses `apply()`'s own defaults.
        cuda_kwargs: forwarded as `**kwargs` to
            `WinCore.precision.cuda_perf_defaults()` (e.g.
            `{"cudnn_benchmark": False}` for a variable-input-shape
            workload -- see that function's docstring for why). `None`
            (default) uses that function's own defaults.
        apply_cuda: if `False`, skip the CUDA tuning step entirely
            (`.cuda` is `None` in the result, not an inert plan) --
            e.g. for a CPU-only preprocessing script where checking for
            a CUDA device isn't even meaningful.

    Returns an `OptimizePlan` with both sub-plans plus a flattened
    `.warnings` list (each sub-plan's own warnings, prefixed so you can
    tell which subsystem raised which one, without having to dig into
    `.cpu.warnings` / `.cuda.warnings` separately unless you want the
    detail).

    Never raises for a missing/unavailable subsystem: no CUDA device
    -> `.cuda` is `cuda_perf_defaults()`'s own inert plan (not skipped,
    unless `apply_cuda=False`); `psutil` missing for OS-level
    priority/affinity -> recorded in `.cpu.warnings`, same as calling
    `WinCore.cpu.apply()` directly (see that function's own `strict=`
    parameter if you want a missing OS-level control to raise instead
    -- pass it via `cpu_kwargs={"strict": True}`).
    """
    from . import cpu as _cpu
    from . import precision as _precision

    cpu_plan = _cpu.apply(**(cpu_kwargs or {}))

    cuda_plan = None
    warnings: List[str] = [f"cpu: {w}" for w in getattr(cpu_plan, "warnings", ())]

    if apply_cuda:
        cuda_plan = _precision.cuda_perf_defaults(**(cuda_kwargs or {}))
        warnings.extend(f"cuda: {w}" for w in cuda_plan.warnings)

    return OptimizePlan(cpu=cpu_plan, cuda=cuda_plan, warnings=warnings)
