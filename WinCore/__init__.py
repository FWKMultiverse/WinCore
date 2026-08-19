"""
WinCore Foundation -- a stability + resource-awareness layer built ON
TOP of PyTorch's own CUDA backend, for training on Windows.

Distribution/PyPI name: "WinCore-Foundation" (`pip install
WinCore-Foundation`). Python import name stays `WinCore` (Python
identifiers can't contain hyphens, and this is the same well-
established pattern as e.g. `beautifulsoup4` on PyPI importing as
`bs4`) -- every example in this docstring and in README.md /
API_REFERENCE.md uses `import WinCore`, which is correct and
unchanged by the distribution rename.

Design principle: this package never reimplements CUDA/cuDNN/cuBLAS --
it uses torch.cuda for the actual GPU compute, the same as any PyTorch
script. What it replaces is everything AROUND that compute that Windows
handles worse than Linux: file-lock-safe checkpointing, torch.compile
without Triton's Windows gap, CPU thread scheduling, real hardware spec
checks, safe dtype selection, and optional hand-written fused CUDA
extension kernels for specific memory-bandwidth-bound ops (compiled via
nvcc, not Triton -- so the Windows/Triton gap doesn't apply to them).

Modules:
  - io / compile: file-lock-safe checkpoint writes, torch.compile with
    automatic eager fallback.
  - cpu: tiered logical-thread reservation heuristic, applied to
    torch/OMP/MKL, plus real OS-level process priority and CPU-affinity
    pinning (`set_priority`, `pin_affinity`) for the parts thread-pool
    sizing structurally cannot reach (see WinCore.cpu module docstring).
    Also `cpu_vendor()` (AMD/Intel detection) and NUMA-aware affinity
    pinning (`numa_node_count`/`numa_node_cpus`) for multi-socket Intel
    and AMD Threadripper/EPYC-class machines.
  - bootstrap: `optimize()` — one call applying `cpu.apply()` then
    `precision.cuda_perf_defaults()` in the order that actually
    matters, returning a combined `OptimizePlan`.
  - accumulate: `GradientAccumulator` — correct loss scaling for
    gradient accumulation, plus DDP `no_sync()` skipping on
    non-boundary micro-steps so an N-step accumulation window pays for
    one gradient all-reduce instead of N.
  - spec: real RAM/VRAM/GPU/temperature readout (via optional psutil /
    pynvml / wmi) and minimum-requirement checks.
  - precision: picks a safe torch dtype (fp32/fp16/bf16) from the GPU's
    actual compute capability, plus `quantize_fp8`/`dequantize_fp8`
    (with optional per-axis scaling for higher-fidelity per-
    row/channel quantization) for compressing tensors to real fp8
    storage, `quantize_fp4`/`dequantize_fp4` for EXPERIMENTAL software-
    only 4-bit storage compression (explicitly not hardware-
    accelerated, not the recommended default — see its own docstring),
    and `safe_cast()` to turn float16's silent overflow-to-inf into an
    explicit warning/error/clip instead.
  - thermal: reads GPU temperature (via pynvml) and can pause a
    training loop above a threshold -- monitoring + a software pause,
    not hardware fan/clock control (see WinCore.thermal docstring).
  - diagnostics: catches silent training problems (NaN/Inf loss, loss
    plateaus, exploding/vanishing gradients) and measures dataloader-
    vs-compute time to flag bottlenecks -- read-only observation, never
    touches your model/optimizer/data.
  - kernels: optional hand-written CUDA extension kernels (fused
    bias+GELU so far), compiled with nvcc, not dependent on Triton.
  - multigpu: picks a working torch.distributed backend for the
    platform (gloo on native Windows, nccl on Linux/WSL2), and
    *measures* real interconnect topology (NVLink vs. PCIe distance
    per GPU pair, via pynvml/NVML or nvidia-smi topo -m) to size the
    DDP bucket from an actual reading instead of a guess, plus a
    per-GPU VRAM balance check -- configures torch.distributed/DDP,
    does not reimplement them.
  - memory: Windows-aware DataLoader worker-count defaults (spawn, not
    fork, on Windows), pinned-memory transfer helper, a VRAM-pressure-
    gated cache guard, `trim_working_set()` to release freed host RAM
    that Windows holds onto in the process's working set longer than
    Linux does, `estimate_worker_ram_multiplier()` explaining the
    Windows-specific per-worker RAM duplication from `spawn` (not
    `fork()`) DataLoader workers, and `PinnedBufferPool` to reuse
    pinned host staging buffers instead of re-pinning every step.
  - cache: LRU disk cache for expensive-to-preprocess dataset samples,
    with a byte budget and Windows-safe atomic writes (via
    WinCore.io.atomic_write) so a killed process never leaves a corrupt
    cache entry, plus `.warm()` to prepopulate many entries
    concurrently ahead of the training loop needing them.
  - kv: generic in-memory/on-device step-state cache (append or
    replace, optional sliding-window eviction, optional fp8
    compression) -- not LLM-attention-specific; the same store works
    for RNN/LSTM hidden state, GNN/GAT node embeddings, PPO rollout
    state, or anything else that carries tensor state across steps.
  - power: `prevent_sleep()` stops Windows from suspending the machine
    mid-run (a real risk for unattended multi-hour training that has
    no user input to keep the OS's idle timer from firing) via the
    real `SetThreadExecutionState` API. `check_tdr_risk()` reads
    Windows' GPU driver watchdog timeout (`TdrDelay`) and explains
    whether a long single CUDA kernel launch (e.g. from
    `WinCore.kernels`) risks being killed and surfacing as a confusing
    `CUDA error: unspecified launch failure` with no indication a
    2-second Windows timer, not a bug, caused it.

What this deliberately is NOT:
  - Not a replacement for PyTorch's CUDA/cuDNN kernels for matmul/conv
    -- those stay exactly as fast (or slow) as your installed torch
    build makes them, unchanged.
  - Not a native fp4 compute backend (4-bit is a quantization scheme
    layered by libraries like bitsandbytes, not a dtype this package
    invents).
  - Not a GPU fan/power/clock controller -- `thermal` reads temperature
    via pynvml and can pause your loop; it cannot touch driver-level
    fan curves or power limits.

Public API:
    from WinCore import atomic_write, safe_compile, should_compile
    from WinCore import cpu, spec, precision, thermal, kernels
"""

import os as _os


def _read_bundled(filename: str, fallback_url: str) -> str:
    path = _os.path.join(_os.path.dirname(__file__), filename)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return (
            f"{filename} was not found alongside the installed package "
            f"(this build may predate bundling it as package-data). "
            f"See {fallback_url} for the full docs."
        )


def docs() -> str:
    """Return the full README.md text (overview, install, quick start).

    ``pip install`` normally does NOT leave a browsable README in
    site-packages -- the README you write only becomes the PyPI project
    page's description (via the wheel's METADATA), it isn't copied in
    as a plain file next to the code. That's standard pip behavior, not
    something specific to this package -- but it means someone who
    installs WinCore and then goes looking in the site-packages folder
    for instructions finds nothing.

    This package copies README.md into the installed package directory
    as data (see ``package-data`` in pyproject.toml) specifically so it
    survives `pip install`, and this function is the easy way to read
    it without knowing where site-packages is:

        python -c "import WinCore; print(WinCore.docs())"

    For a full function-by-function API reference (every parameter,
    every return value) instead of the overview, see `api_reference()`.
    ``help(WinCore)`` also works, for the shortest module-level summary.
    """
    return _read_bundled("README.md", "https://github.com/FWKMultiverse/WinCore-Foundation")


def api_reference() -> str:
    """Return the full API_REFERENCE.md text -- every public function
    and class in WinCore, its parameters, return value, and what it
    actually does. This is the reference for "what does this specific
    argument do", as opposed to `docs()`, which is the overview/quick-
    start pitch.

        python -c "import WinCore; print(WinCore.api_reference())"
    """
    return _read_bundled("API_REFERENCE.md", "https://github.com/FWKMultiverse/WinCore-Foundation")


def changelog() -> str:
    """Return the full CHANGELOG.md text -- what changed in each
    released version, including this one's known limitations.

        python -c "import WinCore; print(WinCore.changelog())"
    """
    return _read_bundled("CHANGELOG.md", "https://github.com/FWKMultiverse/WinCore-Foundation")


from .io import (
    atomic_write,
    AtomicWriteError,
    atomic_torch_save,
    atomic_safetensors_save,
    fast_torch_load,
    fast_safetensors_load,
    load_checkpoint,
)
from .compile import safe_compile, should_compile, SafeCompiled
from .bootstrap import optimize, OptimizePlan
from .accumulate import GradientAccumulator
from . import cpu
from . import spec
from . import precision
from . import thermal
from . import kernels
from . import diagnostics
from . import multigpu
from . import memory
from . import cache
from . import kv
from . import power

__all__ = [
    "docs",
    "api_reference",
    "changelog",
    "atomic_write",
    "AtomicWriteError",
    "atomic_torch_save",
    "atomic_safetensors_save",
    "fast_torch_load",
    "fast_safetensors_load",
    "load_checkpoint",
    "safe_compile",
    "should_compile",
    "SafeCompiled",
    "optimize",
    "OptimizePlan",
    "GradientAccumulator",
    "cpu",
    "spec",
    "precision",
    "thermal",
    "kernels",
    "diagnostics",
    "multigpu",
    "memory",
    "cache",
    "kv",
    "power",
]

__version__ = "0.8.2"
