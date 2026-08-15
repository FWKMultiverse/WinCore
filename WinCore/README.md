# WinCore

A stability + resource-awareness layer built **on top of PyTorch's own CUDA backend**, for training on **Windows**. It never reimplements CUDA/cuDNN/cuBLAS — `torch.cuda` still does the actual GPU compute, exactly as fast as your installed torch build makes it. What WinCore replaces is everything *around* that compute which Windows handles worse than Linux.

## Install

```bash
pip install WinCore              # core only, zero hard dependencies
pip install "WinCore[full]"      # + psutil (RAM), pynvml (NVIDIA VRAM), wmi (GPU name)
```

Works fine inside a `.venv`.

```python
import WinCore
```

## What's actually in here

| Module | What it does | Depends on |
|---|---|---|
| `WinCore.io` | `atomic_write` — file-lock-safe checkpoint writes with retry/backoff | nothing |
| `WinCore.compile` | `safe_compile` — `torch.compile` with automatic eager fallback | torch (lazy) |
| `WinCore.cpu` | `recommended_threads` / `apply` — leaves a small reserve of logical threads free for the OS instead of pinning 100% of them | nothing (torch optional) |
| `WinCore.spec` | `get_system_spec` / `meets_minimum` — reads real RAM/VRAM/GPU info and checks it against your minimums | psutil, pynvml, wmi (all optional) |
| `WinCore.precision` | `recommended_dtype` / `resolve_dtype` — picks fp32/fp16/bf16 based on the GPU's actual compute capability. `amp()` — bundles `torch.autocast` + `GradScaler` using that dtype, so mixed precision is one call instead of hand-wiring both and remembering which dtype needs a scaler | torch (lazy) |
| `WinCore.kernels` | `fused_bias_gelu` — a real, hand-written CUDA extension (compiled with nvcc, not Triton) that fuses bias-add + GELU into one kernel launch, for fp16/bf16/fp32/fp64 natively and fp8 via an upcast bridge | torch (falls back to a plain-PyTorch implementation with a warning if CUDA Toolkit + MSVC Build Tools aren't available to compile the extension) |
| `WinCore.diagnostics` | `TrainingMonitor` — catches NaN/Inf loss, loss plateaus, exploding/vanishing gradients, dataloader-vs-compute bottlenecks, GPU launch/sync stalls (`gpu_timer`, measured via `torch.cuda.Event` — catches idle gaps a plain utilization-percent reading won't show), and cross-signal co-occurrence (`record_signal`) for e.g. temperature/VRAM readings recorded near the same step. `attach_nan_guards(model)` — hooks every submodule's forward/backward so a non-finite value is caught at the *layer* it first appears, not only once it reaches the loss | nothing for loss/plateau/timing/signals; torch (lazy) for gradient norms, `gpu_timer`, and NaN guards |
| `WinCore.multigpu` | `plan_distributed()` / `ddp_kwargs()` — picks a working `torch.distributed` backend for the platform (`gloo` on native Windows, `nccl` on Linux/WSL2). `init_from_env()` — reads the standard `torchrun` env vars and calls `init_process_group()` with that backend plus `torch.cuda.set_device()`, in one call. `detect_topology()` measures NVLink vs. PCIe distance per GPU pair (via `pynvml`/NVML, or `nvidia-smi topo -m` as fallback) and sizes the DDP bucket from that reading instead of a guess. `check_gpu_balance()` flags a lopsided multi-GPU VRAM setup before it OOMs one rank | pynvml or `nvidia-smi` on PATH for real topology (falls back to a GPU-count heuristic, clearly labeled as such, if neither is available); torch (lazy) |
| `WinCore.memory` | `recommended_dataloader_kwargs()` — Windows-aware (`spawn`, not `fork`) worker-count defaults; `CacheGuard` — calls `torch.cuda.empty_cache()` only when VRAM is actually under pressure, not on a fixed schedule | torch (lazy) |
| `WinCore.cache` | `DiskCache` — LRU, byte-budgeted disk cache for expensive-to-preprocess dataset samples, with Windows-safe atomic writes | nothing |

### About `WinCore.kernels`

This is a compiled CUDA kernel, built with `torch.utils.cpp_extension` + `nvcc` — the same mechanism `torch.utils.cpp_extension` is officially meant for — rather than a Python wrapper around `torch.nn.functional.gelu`. It does **not** depend on Triton, so Triton's lack of official Windows support doesn't apply to it. You need the CUDA Toolkit + MSVC Build Tools installed to compile it (`python -m WinCore.kernels.build`).

**What it does:** `F.gelu(x + bias)` normally launches two separate kernels, each reading/writing the full tensor to VRAM. This kernel fuses bias-add + GELU into a single launch, so the intermediate `x + bias` result stays in registers instead of making a round trip to VRAM. It does not touch or replace cuBLAS/cuDNN's matmul/convolution kernels — only this specific elementwise op.

**Dtype support:** float32 / float64 / float16 / bfloat16 all run as genuine single-kernel-launch fused ops (templated in the `.cu` file). float8 (`torch.float8_e4m3fn` / `torch.float8_e5m2`, PyTorch 2.1+) is handled by transparently upcasting to float32, running the fused kernel there, and casting back — correct output, but without the fusion speedup for that specific dtype, since raw fp8 elementwise arithmetic isn't a portable CUDA operation the way fp16/bf16 are (see the `.cu` file header for why). `fp4` remains intentionally unsupported — it's a quantization scheme (e.g. via `bitsandbytes`), not a compute dtype; `precision.resolve_dtype("fp4")` still raises on purpose.

**Toolchain fallback:** if the CUDA Toolkit / ninja / MSVC `cl.exe` aren't available on a given machine, `fused_bias_gelu()` warns once and falls back to an unfused (numerically identical) plain-PyTorch implementation, so the rest of WinCore stays usable on a machine without a full native build toolchain. Check `WinCore.kernels.kernel_status()` to see which path is active.

**Confirmed on a real machine:** all 11 tests in `tests/test_fused_bias_gelu.py` passed on a real Windows + CUDA + MSVC (VS2026, `-allow-unsupported-compiler`) setup, including `test_fused_is_at_least_as_fast_as_unfused_for_large_tensor` — the fused kernel actually compiles and runs correctly there. This is one confirmed machine/GPU/toolchain combination, not a claim about every machine — build and run `pytest tests/test_fused_bias_gelu.py -v` on yours to check.

### `WinCore.thermal`

Reads GPU temperature via `pynvml.nvmlDeviceGetTemperature` (NVIDIA's own helper — this doesn't invent its own sensor reading) and, if it's over a threshold, pauses the training loop for a bit before the next step:

```python
guard = WinCore.thermal.ThermalGuard(threshold_c=83)
for step, batch in enumerate(loader):
    train_step(batch)
    if step % 20 == 0:
        guard.check()   # sleeps if over threshold, otherwise returns instantly
```

This is monitoring + a software-level pause, not hardware control — it can't touch fan curves, power limits, or clocks (those live in the driver/BIOS/vendor overlay, which already throttles independently of this).

Pass `monitor=` a `TrainingMonitor` (see below) and every temperature reading feeds into it automatically — see the cross-signal example below.

### `WinCore.diagnostics`

Catches the "training didn't crash, but something's silently wrong" class of problems, and reports where wall-clock time is actually going — without changing your model, optimizer, or data pipeline. It only observes and calls `on_issue`; you decide what to do about what it finds.

```python
from contextlib import nullcontext
import WinCore

monitor = WinCore.diagnostics.TrainingMonitor(on_issue=lambda i: print(i.severity, i.message))
thermal_guard = WinCore.thermal.ThermalGuard(threshold_c=83, monitor=monitor)

for step, batch in enumerate(loader):
    with monitor.data_timer():
        x, y = batch
    with monitor.compute_timer():
        with monitor.gpu_timer() if step % 20 == 0 else nullcontext():   # periodic sampling; see cost note below
            loss = train_step(x, y)
            loss.backward()
    monitor.record_loss(step, loss.item())          # NaN/Inf + plateau detection
    monitor.record_grad_norm(step, model)            # exploding/vanishing gradients
    thermal_guard.check(step)                        # also feeds temperature into `monitor`
    optimizer.step()
    optimizer.zero_grad()

monitor.bottleneck_report()   # data-vs-compute wall time split (DataLoader bottleneck if >=40%),
                               # plus GPU busy-vs-idle time from any gpu_timer() samples taken
```

`gpu_timer()` calls `event.synchronize()` once per use, which is a real (small) stall — that's why the example above only samples it every 20 steps rather than wrapping every step.

**Cross-signal co-occurrence:** `monitor.record_signal(step, name, value)` accepts a reading from anywhere — GPU temperature (wired automatically via `ThermalGuard(monitor=...)` above), VRAM pressure, your own LR-schedule changes, whatever you want tracked. Any loss/gradient Issue emitted at a nearby step (within `signal_correlation_window`, default 3 steps) is annotated with whichever signals were recorded around that same step, so e.g. a gradient explosion that happened right after a thermal pause shows up together instead of as two log lines you'd have to line up by hand. This is a same-window **co-occurrence** note, not a causal claim — it surfaces signals that were already being recorded, side by side; it doesn't assert the signal caused the issue.

Detects, with real logic (not vibes):
- **NaN/Inf loss** — flagged the step it appears, with likely causes in the message.
- **Loss plateau** — rolling-window relative improvement below a threshold.
- **Exploding gradients** — grad norm jumping by a large factor step-to-step.
- **Vanishing gradients** — grad norm staying near-zero for a sustained streak.
- **DataLoader bottleneck** — if ≥40% of measured wall-clock time is spent waiting on data instead of compute, meaning the GPU is likely idling between batches while the loss still looks fine.
- **GPU launch/sync stalls** (`gpu_timer`) — if ≥20% of the wall-clock time spent inside a `gpu_timer()` block wasn't matched by actual GPU-clock execution time (measured via `torch.cuda.Event`), flags that the GPU spent that time idle inside the block — a gap a coarse utilization-percent reading doesn't show. This identifies *that* time is going missing and roughly *where* in the loop, not *why* — use `torch.profiler` to pin down the specific cause.

The loss/plateau/timer logic needs no dependencies. `record_grad_norm` needs torch (imported lazily) — both paths are now confirmed: the full suite (`tests/test_diagnostics.py` plus the rest of `tests/`) has passed end-to-end on a real Windows machine with torch + CUDA installed (115 passed, 3 skipped — the skips are tests specifically for the no-CUDA fallback path, correctly inert on a machine that does have CUDA).

## New in 0.7.1: fp8 tensor compression, generic step-state cache, OS-level CPU control, Windows RAM tools

`WinCore.precision.quantize_fp8(tensor)` / `dequantize_fp8(packed)` compress a
float16/bfloat16/float32 tensor to real fp8 storage (`float8_e4m3fn` /
`float8_e5m2`) with dynamic per-tensor scaling, for shrinking a KV cache,
activation checkpoints, or optimizer state on Hopper/Ada+ GPUs — distinct
from `amp()`, which picks a *compute* dtype rather than compressing storage.

`WinCore.kv.StepCache` is a generic keyed store for per-step tensor state —
not attention/LLM-specific. The same class works for attention KV
(`mode="append"`, sliding-window eviction via `max_len`), RNN/LSTM hidden
state or GNN/GAT node embeddings (`mode="replace"`), or anything else that
carries tensor state across steps, with optional `compress=True` to keep
everything in fp8 storage automatically. See `WinCore/kv.py` for the full
rationale and API docs.

`WinCore.cpu.set_priority()` / `pin_affinity()` are real OS-scheduler calls
(Win32 `SetPriorityClass` / POSIX `nice`, and `SetProcessAffinityMask` /
`sched_setaffinity`, via `psutil`) — unlike `torch.set_num_threads()`, these
reach the Python main loop and DataLoader worker processes, not just a
single BLAS op's internal thread pool. `apply(priority=..., affinity=...)`
wires both in, non-fatally by default.

`WinCore.memory.trim_working_set()` calls the real Win32
`SetProcessWorkingSetSize(handle, -1, -1)` to release already-freed pages
from the process's working set — fixes the "Task Manager still shows 20GB
RAM long after the big batch/preprocessing step finished and was garbage
collected" symptom, which is a Windows working-set-retention behavior, not
a Python memory leak. `WinCore.memory.estimate_worker_ram_multiplier()`
explains the separate, also-real issue where Windows `spawn`-based
DataLoader workers each get a full copy of an in-memory `Dataset` (unlike
Linux `fork()`'s copy-on-write sharing) — roughly `(num_workers + 1)x` RAM
for whatever the dataset holds in memory.

```python
cache = WinCore.kv.StepCache(max_len=2048, compress=True)
cache.update("layer0.k", new_keys, mode="append")
k = cache.get("layer0.k")  # dequantized transparently

WinCore.cpu.apply(priority="above_normal", affinity=True)

# between epochs / after a big preprocessing pass:
WinCore.memory.trim_working_set()
print(WinCore.memory.estimate_worker_ram_multiplier(num_workers=6))
```

## Quick start

```python
import WinCore

# 1. Leave the OS some headroom, apply to torch/OMP/MKL automatically.
# Pass priority/affinity too if you want the OS-scheduler-level controls
# (these reach plain Python loops and DataLoader workers -- torch thread
# count alone can't; see WinCore.cpu module docstring for why):
plan = WinCore.cpu.apply(priority="above_normal", affinity=True)
print(plan)  # ThreadPlan(total_logical=16, reserved=3, recommended=13)
if plan.warnings:
    print("non-fatal OS-level warnings:", plan.warnings)

# 2. Check the machine actually meets your training script's minimums
check = WinCore.spec.meets_minimum(min_vram_gb=6, min_ram_gb=16)
if not check.ok:
    raise SystemExit(f"Machine doesn't meet requirements: {check.reasons}")

# 3. Pick a safe default dtype for this GPU
dtype = WinCore.precision.recommended_dtype()

# 4. Compile safely (falls back to eager if Triton misbehaves on Windows)
model = WinCore.safe_compile(model)

# 5. Save checkpoints without random WinError 32 crashes
WinCore.atomic_write(lambda p: torch.save(model.state_dict(), p), "checkpoint.pt")
```

### Multi-GPU (2-4+ GPUs)

```python
import WinCore
from torch.nn.parallel import DistributedDataParallel as DDP

plan = WinCore.multigpu.plan_distributed()   # picks gloo on native Windows, nccl on Linux/WSL2
print(plan.reason)                            # explains the backend AND whether the bucket size is measured or a fallback

topo = WinCore.multigpu.detect_topology()
if topo.measured:
    for link in topo.links:
        print(f"GPU{link.gpu_a} <-> GPU{link.gpu_b}: {link.label}")
else:
    print(topo.note)  # honest about why (no pynvml/nvidia-smi) instead of guessing

balance = WinCore.multigpu.check_gpu_balance()
if balance.warning:
    print(balance.warning)  # e.g. one card also driving a display, less free VRAM

# Launched via `torchrun --nproc_per_node=N your_script.py` (or mp.spawn setting the same env vars);
# init_from_env() reads RANK/WORLD_SIZE/LOCAL_RANK, calls init_process_group with `plan`'s backend,
# and puts this process on the right GPU:
plan = WinCore.multigpu.init_from_env(plan)
model = DDP(model, **WinCore.multigpu.ddp_kwargs(plan, find_unused_parameters=False))
```

### Mixed precision, without hand-wiring autocast + GradScaler

```python
import WinCore

ctx = WinCore.precision.amp()   # picks dtype from recommended_dtype(), scaler on only for fp16
print(ctx.plan.reason)

for step, (x, y) in enumerate(loader):
    optimizer.zero_grad()
    with ctx.autocast():
        loss = model(x, y)
    ctx.scaler.scale(loss).backward()
    ctx.scaler.step(optimizer)
    ctx.scaler.update()
```

On a GPU where `recommended_dtype()` resolves to `float32` (or with no CUDA device at all), `ctx.autocast()` and `ctx.scaler` are both real no-ops — the loop above doesn't need an `if` branch for that case.

### Data pipeline: DataLoader defaults + SSD cache

```python
import WinCore
from torch.utils.data import DataLoader

dl_plan = WinCore.memory.recommended_dataloader_kwargs()
loader = DataLoader(dataset, batch_size=32,
                     num_workers=dl_plan.num_workers,
                     pin_memory=dl_plan.pin_memory,
                     persistent_workers=dl_plan.persistent_workers,
                     prefetch_factor=dl_plan.prefetch_factor)

cache = WinCore.cache.DiskCache("D:/wincore_cache", max_bytes=20 * 1024**3)

class MyDataset(torch.utils.data.Dataset):
    def __getitem__(self, idx):
        return cache.get_or_compute(idx, lambda: self._load_and_preprocess(idx))
```

### Background-level NaN detection (not just the loss)

```python
import WinCore

guard = WinCore.diagnostics.attach_nan_guards(
    model, on_issue=lambda i: print(i.severity, i.data["module"], i.message)
)
# ... train for a bit, or until the issue you're chasing reproduces ...
guard.detach()  # or use `with WinCore.diagnostics.attach_nan_guards(model) as guard: ...`
```

### VRAM-pressure-aware cache clearing

```python
import WinCore

cache_guard = WinCore.memory.CacheGuard(min_free_fraction=0.10)
for step, batch in enumerate(loader):
    train_step(batch)
    if step % 50 == 0:
        cache_guard.check()  # only calls torch.cuda.empty_cache() if actually under pressure
```

## What this is *not* (on purpose)

Being upfront about scope, because overselling this would be worse than not building it:

- **Not faster than PyTorch's own CUDA/cuDNN kernels.** This package calls into `torch`; it does not reimplement or beat NVIDIA's own kernel libraries. No Python-level wrapper can.
- **Not a profiler.** `gpu_timer()` tells you how much wall-clock time inside a block wasn't matched by GPU-clock execution time; it does not tell you which line or kernel caused that gap. For that, use `torch.profiler`.
- **`amp()` doesn't implement mixed precision.** It picks a dtype and constructs `torch.autocast` + `torch.amp.GradScaler` with it — the actual numerics are entirely `torch`'s.
- **Not a GPU thermal controller.** Python has no portable access to that layer. `spec` can *read* GPU info if `pynvml` is installed; it cannot throttle or control hardware.
- **Not a native fp4 compute type.** 4-bit is a quantization scheme (e.g. via `bitsandbytes`), a different technique from picking a compute dtype. `precision.resolve_dtype("fp4")` raises `ValueError` on purpose rather than silently pretending to support it.
- **No CPU-generation gate.** There's no reliable, portable way to detect "i5-9400f or newer" from Python — CPU name strings aren't standardized enough. `meets_minimum()` checks logical thread count and VRAM/RAM instead, which are actually measurable; an optional advisory `cpu_name_contains` substring check is available but explicitly not authoritative.
- **No GNN-specific kernels or graph-batching logic.** `WinCore.multigpu`, `.memory`, `.precision`, and `.diagnostics` all work the same whether your model is a GNN, a transformer, or a CNN — none of that is architecture-specific, and none of it requires calling every module (use whichever pieces are useful; nothing here is required by another WinCore module or by torch itself). There's no PyG/DGL-specific sparse-batching or message-passing code, and this package doesn't claim to have any.

## Recommended minimum hardware (informational, not enforced)

These are advisory baselines, not a hard gate `meets_minimum()`
enforces — see "No CPU-generation gate" above for why CPU-generation
detection specifically stays advisory-only (an optional
`cpu_name_contains` substring check exists but isn't authoritative).
GPU compute capability, unlike a CPU name string, *is* reliably
readable, so `spec.meets_minimum()` can enforce that part for real.

- **NVIDIA:** GTX 1060 6GB or newer (up through current-gen)
- **AMD:** RX 580 8GB or newer (up through current-gen)
- **Intel CPU:** i5-8500 / i3 10th-gen or newer (up through current-gen)
- **AMD CPU:** Ryzen 5 3600 or newer (up through current-gen)

Below these, WinCore will generally still run — you'll just be more
likely to hit VRAM/RAM ceilings or dtype fallbacks (e.g. no bf16
support pre-Ampere) sooner.

## Development

```bash
pip install -e ".[dev]"
pytest
```
