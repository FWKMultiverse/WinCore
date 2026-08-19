"""
WinCore.kernels — real, compiled CUDA extension kernels (not Triton).

Why Triton isn't used here
---------------------------
torch.compile's default Inductor backend depends on Triton, which has no
official Windows support (see WinCore.compile docstring). This package
sidesteps that entirely by writing raw CUDA C++ (.cu) compiled with nvcc
via `torch.utils.cpp_extension` -- the same mechanism NVIDIA's own
apex/Megatron-LM fused kernels use, and one of the officially supported
ways to extend PyTorch. nvcc + a CUDA Toolkit install work the same way
on Windows as on Linux, once MSVC Build Tools (cl.exe) are installed
alongside it -- Triton's Windows gap does not apply here.

What's genuinely here vs. what's not
-------------------------------------
`fused_bias_gelu` is a real, hand-written, compilable CUDA kernel that
fuses a bias-add and GELU activation into a single kernel launch, saving
one full VRAM round-trip vs. calling them as separate PyTorch ops. This
is a legitimate, narrow optimization technique (kernel fusion for
memory-bandwidth-bound elementwise ops) -- it is not a claim to beat
cuBLAS/cuDNN's compute-bound matmul/convolution kernels, which remain
the fastest available and are not touched here.

This module has NOT been compiled or benchmarked in the sandbox this
was written in -- there is no GPU there. `build.py` in this folder
compiles it via nvcc; you need a Windows (or Linux) machine with a CUDA
Toolkit and a working C++ compiler (MSVC Build Tools on Windows) to
build and actually benchmark it. Nothing in this package pretends
otherwise.
"""

# BUGFIX (v0.8.2, was #8.1): this used to be an eager
# `from .fused_bias_gelu import (...)` at the top of this file. That
# module imports `torch` at ITS top level too, so the moment anything
# did `import WinCore` -- which runs `from . import kernels` in
# WinCore/__init__.py -- torch was already imported and fully
# initialized (OMP/MKL thread pools included) before user code ever
# got a chance to run. That silently broke WinCore.cpu.apply()'s
# documented contract ("call this before anything imports torch"):
# apply() would set OMP_NUM_THREADS/MKL_NUM_THREADS via
# `os.environ.setdefault(...)`, but those only affect *future* torch/
# OMP initialization -- by the time apply() ran, initialization had
# already happened using whatever the OS/shell already had set (or
# the OMP/MKL default, usually "use every logical core"), no matter
# how early in the caller's script `apply()` was invoked.
#
# Fix: make this subpackage's own re-export lazy via PEP 562 module
# `__getattr__`, so `import WinCore` / `from . import kernels` only
# imports THIS file (no torch touched) -- the kernel submodule, and
# the `import torch` inside it, is deferred until the caller actually
# reaches into `WinCore.kernels.fused_bias_gelu(...)`,
# `WinCore.kernels.FusedBiasGELU(...)`, etc. A caller who does
# `WinCore.cpu.apply()` first (as documented) and only then touches
# `WinCore.kernels.*` now gets the ordering they were promised.
#
# BUGFIX #2 (found via a real-hardware test run after 0.8.2's initial
# release; see CHANGELOG): the fix above wasn't the whole story. Two
# further, related bugs surfaced once this was actually exercised on
# real hardware in a full test-suite run (not just in isolation):
#
#   (a) `__getattr__` below used to bind ONLY the one name it was
#       asked to resolve (`globals()[name] = value`) before returning.
#       If some OTHER lazy name (`kernel_status`, `FusedBiasGELU`,
#       etc.) was resolved FIRST -- before `fused_bias_gelu` itself
#       was ever accessed -- the `importlib.import_module()` call
#       needed to reach that other name ALSO auto-binds the submodule
#       onto `globals()["fused_bias_gelu"]` as a side effect (see (b)
#       below for why), and the old code never corrected that binding
#       since it only touched the name it was actually asked to
#       resolve. Fixed by binding every lazy name that maps to the
#       same submodule in the SAME call, regardless of which one
#       triggered it.
#
#   (b) That fix alone was still not sufficient: Python's import
#       system binds an imported submodule onto its parent package's
#       namespace under the submodule's own name as an UNAVOIDABLE
#       side effect of ANY import of that submodule -- not only ones
#       going through this file's `__getattr__`. A direct import
#       elsewhere in the codebase (a test file needing a private
#       helper not exposed through the lazy public names, e.g.) would
#       independently trigger the exact same corruption, with
#       `__getattr__` never even being called. The submodule
#       previously being named `fused_bias_gelu.py` -- IDENTICAL to
#       the public function name `WinCore.kernels.fused_bias_gelu` --
#       made this collision possible at all. Fixed at the root: the
#       file is now `_fused_bias_gelu_kernel.py` (see that file's own
#       docstring for the full story), so any direct import of it
#       binds a *different* name than the public `fused_bias_gelu`
#       attribute, and the two can never collide again.
#
# `python -m WinCore.kernels.build` (direct submodule import of
# `build`, a name that was never part of this collision) is
# unaffected by any of this either way.

_LAZY_EXPORTS = {
    "fused_bias_gelu": "_fused_bias_gelu_kernel",
    "FusedBiasGELU": "_fused_bias_gelu_kernel",
    "kernel_status": "_fused_bias_gelu_kernel",
    "KernelStatus": "_fused_bias_gelu_kernel",
    "current_fusion_threshold": "_fused_bias_gelu_kernel",
    "calibrate_fusion_threshold": "_fused_bias_gelu_kernel",
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str):
    """PEP 562 module-level lazy attribute access. Only runs on the
    FIRST access of any of these names (Python caches the results in
    this module's globals afterward via the assignment below, so this
    function body doesn't re-run on subsequent accesses -- no repeated
    import overhead).

    BUGFIX (found via a real-hardware test run after 0.8.2's initial
    release -- see CHANGELOG): this used to only bind the ONE name
    that triggered this specific call (`globals()[name] = value`) and
    return it, leaving every OTHER name this submodule provides
    unbound until its own turn came. That was broken specifically for
    the name `"fused_bias_gelu"`, because it's spelled identically to
    ITS OWN submodule (`WinCore/kernels/fused_bias_gelu.py`) -- and
    `importlib.import_module()` has an unavoidable side effect: Python
    ALWAYS auto-binds an imported submodule onto its parent package's
    namespace under the submodule's own name, the same mechanism that
    makes `import package.submodule; package.submodule` work at all.
    So calling `importlib.import_module(".fused_bias_gelu", ...)` to
    resolve some OTHER name first (e.g. `kernel_status`, `FusedBiasGELU`)
    silently set `globals()["fused_bias_gelu"]` to the SUBMODULE
    (module object) as a side effect of that import -- and since this
    function only overwrote the ONE name it was actually asked to
    resolve that call (`kernel_status`, not `fused_bias_gelu`), the
    submodule's auto-binding was never corrected. Once a name is
    present in `globals()`, Python's attribute lookup never calls
    `__getattr__` again for it -- so `WinCore.kernels.fused_bias_gelu`
    would permanently resolve to the module (not the function) for
    the rest of the process, as soon as ANY of the other four lazy
    names (`kernel_status`, `KernelStatus`, `FusedBiasGELU`,
    `current_fusion_threshold`, `calibrate_fusion_threshold`) was
    accessed even once before `fused_bias_gelu` itself was. Confirmed
    with a minimal reproduction, and this was the exact cause of
    `TypeError: 'module' object is not callable` when running the full
    test suite together (order-dependent: an earlier test touching
    `WinCore.kernels.build`, which checks `kernel_status()`, poisoned
    `fused_bias_gelu` for every test file after it) but NOT when
    running `test_fused_bias_gelu.py` alone (nothing else had touched
    the other four names first in that run).

    Fix: whichever name triggers this call, bind EVERY lazy name that
    maps to the same submodule right now, not just the one requested
    -- so the very first access to any of the five, in any order,
    fully and correctly resolves all five before returning.
    """
    submodule_name = _LAZY_EXPORTS.get(name)
    if submodule_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    try:
        submodule = importlib.import_module(f".{submodule_name}", __name__)
    except ImportError:
        # Not built yet (no compiled extension present), or torch
        # itself isn't installed. Accessing one of these names should
        # raise a clear AttributeError, not crash `import WinCore`
        # (which no longer touches this code path at all) or the
        # whole `WinCore.kernels` namespace.
        raise AttributeError(
            f"WinCore.kernels.{name} is unavailable: {submodule_name} "
            f"could not be imported (missing torch, or the compiled "
            f"extension isn't built yet -- run "
            f"`python -m WinCore.kernels.build`)."
        ) from None

    # Bind every name this submodule provides, right now -- not just
    # `name`. This overwrites the import system's own auto-binding of
    # the submodule itself onto `fused_bias_gelu` (see the docstring
    # above) with the real function, in the SAME call that resolves
    # any of its four siblings, regardless of which one was requested
    # first.
    for exported_name, mapped_submodule in _LAZY_EXPORTS.items():
        if mapped_submodule == submodule_name:
            globals()[exported_name] = getattr(submodule, exported_name)

    return globals()[name]


def __dir__():
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
