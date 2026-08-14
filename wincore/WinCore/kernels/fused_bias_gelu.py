"""
Python-side wrapper around the compiled fused_bias_gelu CUDA extension.

Import of the compiled extension is lazy and happens on first use, not
at package-import time -- so `import WinCore` never fails just because
this optional kernel hasn't been built yet on this machine.

Build-toolchain fallback
-------------------------
Compiling this kernel needs the CUDA Toolkit + ninja + a working MSVC
`cl.exe` on PATH (Windows) -- on *any* machine missing one of those,
`build()` used to raise straight out of `fused_bias_gelu()`, which
meant a teammate who just wants to try the rest of WinCore couldn't,
without first setting up a full native build toolchain on their own
machine. That's a real usability problem for "share this with a group
to test," not a hard requirement of the technique itself: the fused
kernel is a bandwidth optimization on top of math that plain PyTorch
already implements correctly (`F.gelu(x + bias)`), so a same-numerics,
pure-PyTorch fallback is always available.

`fused_bias_gelu()` now tries the compiled extension once, and if that
build fails for an environment reason (missing ninja/cl.exe/CUDA
Toolkit, or a compile error), it warns once and transparently falls
back to the unfused PyTorch path for the rest of the process -- same
inputs/outputs/autograd behavior, just without the single-kernel-launch
speedup. Call `kernel_status()` to check which path is actually active
(e.g. before relying on the speed win in a benchmark).

Overhead-aware dispatch (small tensors)
-----------------------------------------
The fused kernel is a *bandwidth* optimization: it wins by avoiding an
extra VRAM round-trip for large elementwise tensors. For small tensors
that round-trip is cheap to begin with, and the fixed cost of the
custom kernel launch (plus the Python -> C++ extension call itself)
can outweigh what fusion saves -- this isn't a hypothetical, it's what
the first real benchmark run against this file showed on an actual
RTX 3060 (reference 0.0300 ms vs fused 0.0664 ms -- fused was *slower*,
0.45x -- for the small tensor size that test used). Below
`_min_elements_for_fusion()` elements, `fused_bias_gelu()` now skips
the extension call entirely and runs the plain-PyTorch path directly
(still correct, still autograd-compatible -- just not the single-
kernel-launch version), instead of always paying for the fused path
regardless of whether it's actually worth it.

The default threshold (`_DEFAULT_MIN_ELEMENTS_FOR_FUSION`) is a
starting heuristic, not a hardware-measured constant -- kernel launch
overhead vs. memory bandwidth varies by GPU generation, driver, and
CUDA Toolkit version, none of which can be honestly benchmarked from a
sandbox with no GPU. Two ways to set it for real instead of guessing:

  1. `calibrate_fusion_threshold()` -- actually benchmarks both paths
     at increasing sizes on THIS machine and finds the real crossover
     point. Requires a CUDA GPU and the compiled extension; run it
     once per machine (or once per GPU model you plan to support) and
     the result applies for the rest of the process.
  2. The `WINCORE_FUSED_MIN_ELEMENTS` environment variable, if you'd
     rather set a known-good value directly (e.g. from a value
     `calibrate_fusion_threshold()` already gave you on a matching
     GPU) without re-running the benchmark every process start.

`_should_use_fused_kernel()` is a small pure function (no torch
dependency) specifically so this decision logic has direct unit
coverage without needing a GPU -- see tests/test_fused_bias_gelu.py.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass

import torch
import torch.nn.functional as F

_ext = None
_ext_unavailable_reason: str | None = None
_warned = False

# Starting heuristic, not a measured constant -- see module docstring
# ("Overhead-aware dispatch") for why this can't be honestly tuned
# without a real GPU, and how to calibrate it for real on yours.
_DEFAULT_MIN_ELEMENTS_FOR_FUSION = 1_048_576  # 1M elements

_calibrated_min_elements: int | None = None  # set by calibrate_fusion_threshold()


@dataclass(frozen=True)
class KernelStatus:
    """Which implementation `fused_bias_gelu` is actually using right now."""

    backend: str  # "cuda_extension" | "pytorch_fallback" | "not_yet_used"
    reason: str | None = None  # why it's on fallback, if it is


def kernel_status() -> KernelStatus:
    """Inspect which backend is (or would be) used, without forcing a
    build. Useful in benchmarks/tests to avoid asserting a speedup that
    can't be true on the fallback path."""
    if _ext is not None:
        return KernelStatus("cuda_extension")
    if _ext_unavailable_reason is not None:
        return KernelStatus("pytorch_fallback", _ext_unavailable_reason)
    return KernelStatus("not_yet_used")


def _get_ext():
    """Build (once) and cache the compiled extension. Returns None if
    the build toolchain isn't available on this machine -- callers fall
    back to plain PyTorch ops instead of crashing."""
    global _ext, _ext_unavailable_reason, _warned
    if _ext is not None or _ext_unavailable_reason is not None:
        return _ext

    from .build import build

    try:
        _ext = build()
    except (RuntimeError, OSError, ImportError) as exc:
        # RuntimeError: our own _check_ninja()/_check_cl() messages, or
        #   an nvcc/MSVC compile failure surfaced by torch.
        # OSError/ImportError: e.g. the compiled .pyd failed to load
        #   (missing CUDA runtime DLLs on PATH for this machine).
        _ext_unavailable_reason = str(exc)
        if not _warned:
            warnings.warn(
                "WinCore.kernels.fused_bias_gelu: couldn't build/load the "
                "compiled CUDA kernel on this machine, so falling back to "
                "an unfused (but numerically identical) PyTorch "
                "implementation -- correct results, without the single-"
                "kernel-launch speedup. Original error:\n"
                f"{_ext_unavailable_reason}",
                RuntimeWarning,
                stacklevel=3,
            )
            _warned = True
        return None
    return _ext


def _unfused(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return F.gelu(x + bias, approximate="tanh")


def _should_use_fused_kernel(numel: int, min_elements: int) -> bool:
    """Pure decision function (no torch dependency) so this logic has
    direct unit coverage without needing a GPU. `numel` is the input
    tensor's element count; `min_elements` is the current threshold
    from `_min_elements_for_fusion()`."""
    return numel >= min_elements


def _min_elements_for_fusion() -> int:
    """Resolution order: a value set by `calibrate_fusion_threshold()`
    this process (highest priority -- it's an actual on-machine
    measurement) > the `WINCORE_FUSED_MIN_ELEMENTS` env var > the
    conservative untuned default."""
    if _calibrated_min_elements is not None:
        return _calibrated_min_elements
    env = os.environ.get("WINCORE_FUSED_MIN_ELEMENTS")
    if env:
        try:
            return int(env)
        except ValueError:
            warnings.warn(
                f"WINCORE_FUSED_MIN_ELEMENTS={env!r} isn't a valid integer; "
                f"ignoring it and using the default threshold instead.",
                RuntimeWarning,
            )
    return _DEFAULT_MIN_ELEMENTS_FOR_FUSION


def current_fusion_threshold() -> int:
    """The element-count threshold `fused_bias_gelu` is currently using
    to decide whether to call the compiled kernel at all -- see
    `_min_elements_for_fusion()` for where it comes from."""
    return _min_elements_for_fusion()


def calibrate_fusion_threshold(
    num_features: int = 4096,
    max_rows: int = 8192,
    trials: int = 15,
    warmup: int = 5,
) -> int:
    """Actually benchmark the fused vs. unfused path on THIS machine,
    at increasing tensor sizes, and find the real crossover point where
    the fused kernel starts winning -- then set that as the threshold
    `fused_bias_gelu()` uses for the rest of this process.

    Requires a CUDA GPU and a machine that can build the extension
    (see `WinCore.kernels.build`); raises `RuntimeError` if either is
    missing rather than silently returning an unmeasured guess -- an
    uncalibrated number pretending to be a measurement would be worse
    than just keeping the documented default.

    This does real GPU work (`trials` timed calls at each of several
    row counts, doubling from a small size up to `max_rows`) and can
    take a few seconds. Run it once per machine/GPU model you plan to
    support, not on every process start -- cache the returned value
    (e.g. via `WINCORE_FUSED_MIN_ELEMENTS`) if startup latency matters.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "calibrate_fusion_threshold() needs a CUDA GPU to produce a "
            "real measurement -- none is available on this machine."
        )
    if _get_ext() is None:
        raise RuntimeError(
            "calibrate_fusion_threshold() needs the compiled CUDA extension "
            f"to be buildable on this machine, and it isn't: "
            f"{_ext_unavailable_reason}"
        )

    import time

    global _calibrated_min_elements

    device = torch.device("cuda")
    rows = 64
    crossover = _DEFAULT_MIN_ELEMENTS_FOR_FUSION
    found = False

    while rows <= max_rows:
        x = torch.randn(rows, num_features, device=device, dtype=torch.float32)
        bias = torch.randn(num_features, device=device, dtype=torch.float32)

        for _ in range(warmup):
            _unfused(x, bias)
            _FusedBiasGELUFunction.apply(x, bias)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(trials):
            _unfused(x, bias)
        torch.cuda.synchronize()
        unfused_ms = (time.perf_counter() - t0) * 1000 / trials

        t0 = time.perf_counter()
        for _ in range(trials):
            _FusedBiasGELUFunction.apply(x, bias)
        torch.cuda.synchronize()
        fused_ms = (time.perf_counter() - t0) * 1000 / trials

        if fused_ms <= unfused_ms:
            crossover = rows * num_features
            found = True
            break
        rows *= 2

    _calibrated_min_elements = crossover
    if not found:
        warnings.warn(
            f"calibrate_fusion_threshold(): fused kernel never beat the "
            f"unfused path up to {max_rows}x{num_features} elements on "
            f"this machine -- using the largest size tested "
            f"({crossover} elements) as the threshold, but consider "
            f"raising max_rows to actually find the crossover, or this "
            f"GPU/build may just not benefit from this specific fusion.",
            RuntimeWarning,
        )
    return crossover


_FP8_DTYPE_NAMES = ("float8_e4m3fn", "float8_e5m2")


def _is_fp8(dtype: torch.dtype) -> bool:
    return any(
        hasattr(torch, name) and dtype == getattr(torch, name)
        for name in _FP8_DTYPE_NAMES
    )


class _FusedBiasGELUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bias):
        ext = _get_ext()
        ctx.save_for_backward(x, bias)
        return ext.fwd(x, bias)

    @staticmethod
    def backward(ctx, grad_out):
        x, bias = ctx.saved_tensors
        ext = _get_ext()
        grad_x = ext.bwd(grad_out, x, bias)
        # bias gradient is the sum of grad_x over all dims except the
        # last one (broadcast rule for the bias-add)
        grad_bias = grad_x.reshape(-1, grad_x.size(-1)).sum(dim=0)
        return grad_x, grad_bias


def fused_bias_gelu(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Fused (x + bias) -> GELU, forward and backward.

    Native, single-kernel-launch fusion for float32 / float64 / float16
    / bfloat16 CUDA tensors (see the .cu file for the dtype-templated
    kernel). For float8 (e4m3/e5m2) tensors this transparently upcasts
    to float32, runs the fused kernel there, and casts the result back
    to fp8 -- correct output, but WITHOUT the fusion speedup for that
    dtype specifically (see the .cu file header for why fp8 isn't
    templated in directly). Autograd flows through the cast either way,
    so `.backward()` works normally regardless of dtype.

    Falls back to an unfused (but numerically identical) plain-PyTorch
    implementation in three cases, all without crashing: (1) the CUDA
    extension can't be built/loaded on this machine at all -- with a
    one-time warning; (2) `x` is smaller than the current fusion
    threshold (see "Overhead-aware dispatch" in the module docstring)
    -- silently, since this is an expected, routine choice, not a
    degraded state; (3) never, for fp8, which always routes through
    fusion via the upcast bridge regardless of size, since the upcast
    itself already costs a full extra pass either way.

    Check `kernel_status()` to see whether the extension is available
    at all on this machine, and `current_fusion_threshold()` /
    `calibrate_fusion_threshold()` to see or tune the size-based
    dispatch."""
    if _is_fp8(x.dtype):
        out_f32 = fused_bias_gelu(x.to(torch.float32), bias.to(torch.float32))
        return out_f32.to(x.dtype)
    if _get_ext() is None:
        return _unfused(x, bias)
    if not _should_use_fused_kernel(x.numel(), _min_elements_for_fusion()):
        return _unfused(x, bias)
    return _FusedBiasGELUFunction.apply(x, bias)


__all__ = [
    "fused_bias_gelu",
    "FusedBiasGELU",
    "kernel_status",
    "KernelStatus",
    "current_fusion_threshold",
    "calibrate_fusion_threshold",
]


class FusedBiasGELU(torch.nn.Module):
    """Drop-in module: replaces `x = x + self.bias; x = F.gelu(x)`."""

    def __init__(self, num_features: int):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fused_bias_gelu(x, self.bias)
