# WinCore Foundation

A stability + resource-awareness layer built **on top of PyTorch's own CUDA backend**, for training on **Windows**. It never reimplements CUDA/cuDNN/cuBLAS — `torch.cuda` still does the actual GPU compute, exactly as fast as your installed torch build makes it. What WinCore Foundation replaces is everything *around* that compute which Windows handles worse than Linux.

> **Naming note:** the PyPI/distribution name is `WinCore-Foundation`. The Python import name is unchanged: `import WinCore` — Python identifiers can't contain hyphens, and this is the same well-established split PyPI already uses elsewhere (e.g. `beautifulsoup4` installs, `bs4` imports). Every example below is copy-paste accurate against the actual code.

## Install

```bash
pip install WinCore-Foundation              # core only, zero hard dependencies
pip install "WinCore-Foundation[full]"      # + psutil (RAM), pynvml (NVIDIA VRAM), wmi (GPU name), safetensors (checkpoint format)
```

Works fine inside a `.venv`.

```python
import WinCore
```

## What's actually in here

| Module | What it does | Depends on |
|---|---|---|
| `WinCore.io` | `atomic_write` — file-lock-safe checkpoint writes with retry/backoff. `atomic_torch_save`/`atomic_safetensors_save` — format-specific wrappers with the same guarantee. `fast_torch_load`/`fast_safetensors_load`/`load_checkpoint` — the read-side counterpart: `mmap=True` loading (torch>=2.1) with automatic fallback, format auto-dispatch by extension | nothing (`safetensors` optional for that format) |
| `WinCore.compile` | `safe_compile` — `torch.compile` with automatic eager fallback | torch (lazy) |
| `WinCore.cpu` | `recommended_threads`/`apply` — leaves a small reserve of logical threads free for the OS instead of pinning 100%. `cpu_vendor()` — AMD/Intel detection. `numa_node_count`/`numa_node_cpus` — NUMA-aware affinity pinning for multi-socket Intel Xeon and AMD Threadripper/EPYC machines (`apply(affinity=True, numa_aware=True)`, on by default) | nothing (torch optional; `psutil` for priority/affinity) |
| `WinCore.bootstrap` | `optimize()` — one call applying `cpu.apply()` then `precision.cuda_perf_defaults()` in the order that actually matters, returning a combined `OptimizePlan` | nothing itself (forwards to `cpu`/`precision`) |
| `WinCore.spec` | `get_system_spec`/`meets_minimum` — reads real RAM/VRAM/GPU info and checks it against your minimums | psutil, pynvml, wmi (all optional) |
| `WinCore.precision` | `recommended_dtype`/`resolve_dtype` — picks fp32/fp16/bf16 from the GPU's actual compute capability. `amp()` — bundles `torch.autocast` + `GradScaler`. `cuda_perf_defaults()` — one call for TF32/cuDNN-benchmark/SDPA-backend tuning, capability-checked. `quantize_fp8`/`dequantize_fp8` — fp8 storage compression, now with optional `axis=` per-channel scaling. `quantize_fp4`/`dequantize_fp4` — EXPERIMENTAL software-only 4-bit storage compression (not hardware-accelerated, not the recommended default — see its own docstring). `safe_cast()` — turns float16's silent overflow-to-`inf` into an explicit warning/error/clip | torch (lazy) |
| `WinCore.accumulate` | `GradientAccumulator` — correct loss scaling for gradient accumulation, plus DDP `no_sync()` skipping on non-boundary micro-steps so an N-step window pays for one gradient all-reduce instead of N | nothing (duck-types against `model.no_sync()`) |
| `WinCore.kernels` | `fused_bias_gelu` — a real, hand-written CUDA extension (compiled with nvcc, not Triton) that fuses bias-add + GELU into one kernel launch, for fp16/bf16/fp32/fp64 natively and fp8 via an upcast bridge | torch (falls back to a plain-PyTorch implementation with a warning if CUDA Toolkit + MSVC Build Tools aren't available to compile the extension) |
| `WinCore.diagnostics` | `TrainingMonitor` — catches NaN/Inf loss, loss plateaus vs. regressions (distinct codes/messages), exploding/vanishing gradients, dataloader-vs-compute bottlenecks, GPU launch/sync stalls (`gpu_timer`), cross-signal co-occurrence (`record_signal`), and training phase/ETA tracking (`record_step_time` — `warmup`/`steady_state`/`stalled` classification, model-size-aware warmup heuristic, steady-state-only ETA). `attach_nan_guards(model)` — hooks every submodule's forward/backward so a non-finite value is caught at the *layer* it first appears | nothing for loss/plateau/timing/signals/phase-tracking; torch (lazy) for gradient norms, `gpu_timer`, and NaN guards |
| `WinCore.multigpu` | `plan_distributed()`/`ddp_kwargs()` — picks a working `torch.distributed` backend for the platform. `init_from_env()` — reads `torchrun` env vars and calls `init_process_group()`. `detect_topology()` measures NVLink vs. PCIe distance per GPU pair. `check_gpu_balance()` flags a lopsided multi-GPU VRAM setup | pynvml or `nvidia-smi` on PATH for real topology (falls back to a labeled heuristic); torch (lazy) |
| `WinCore.memory` | `recommended_dataloader_kwargs()` — Windows-aware (`spawn`, not `fork`) worker-count defaults. `CacheGuard` — calls `torch.cuda.empty_cache()` only under real VRAM pressure. `trim_working_set()` — releases freed host RAM Windows holds onto longer than Linux. `PinnedBufferPool` — reuses pinned (page-locked) CPU staging buffers by `(shape, dtype)` instead of re-pinning (a real CUDA-driver-call cost) every step | torch (lazy) |
| `WinCore.cache` | `DiskCache` — LRU, byte-budgeted disk cache for expensive-to-preprocess dataset samples, with Windows-safe atomic writes and cross-process budget enforcement. `.warm(keys, compute_fn, max_workers=)` — prepopulates many entries concurrently via a thread pool, ahead of the training loop needing them | nothing |
| `WinCore.power` | `prevent_sleep()` — stops Windows from suspending the machine mid-run, via `SetThreadExecutionState`. `check_tdr_risk()` — reads Windows' GPU driver watchdog timeout (`TdrDelay`) and explains the risk of a long single CUDA kernel launch getting killed | nothing (stdlib `ctypes`/`winreg` only) |

### About `WinCore.kernels`

This is a compiled CUDA kernel, built with `torch.utils.cpp_extension` + `nvcc` — the same mechanism `torch.utils.cpp_extension` is officially meant for — rather than a Python wrapper around `torch.nn.functional.gelu`. It does **not** depend on Triton, so Triton's lack of official Windows support doesn't apply to it. You need the CUDA Toolkit + MSVC Build Tools installed to compile it (`python -m WinCore.kernels.build`).

**What it does:** `F.gelu(x + bias)` normally launches two separate kernels, each reading/writing the full tensor to VRAM. This kernel fuses bias-add + GELU into a single launch, so the intermediate `x + bias` result stays in registers instead of making a round trip to VRAM. It does not touch or replace cuBLAS/cuDNN's matmul/convolution kernels — only this specific elementwise op.

**Dtype support:** float32 / float64 / float16 / bfloat16 all run as genuine single-kernel-launch fused ops (templated in the `.cu` file). float8 (`torch.float8_e4m3fn` / `torch.float8_e5m2`, PyTorch 2.1+) is handled by transparently upcasting to float32, running the fused kernel there, and casting back — correct output, but without the fusion speedup for that specific dtype, since raw fp8 elementwise arithmetic isn't a portable CUDA operation the way fp16/bf16 are (see the `.cu` file header for why). `fp4` remains intentionally unsupported — it's a quantization scheme (e.g. via `bitsandbytes`), not a compute dtype; `precision.resolve_dtype("fp4")` still raises on purpose.

**Toolchain fallback:** if the CUDA Toolkit / ninja / MSVC `cl.exe` aren't available on a given machine, `fused_bias_gelu()` warns once and falls back to an unfused (numerically identical) plain-PyTorch implementation, so the rest of WinCore stays usable on a machine without a full native build toolchain. Check `WinCore.kernels.kernel_status()` to see which path is active.

**Confirmed on a real machine:** all 11 tests in `tests/test_fused_bias_gelu.py` pass on a real Windows + CUDA + MSVC (VS2026, `-allow-unsupported-compiler`) setup, including `test_fused_is_at_least_as_fast_as_unfused_for_large_tensor` — the fused kernel actually compiles and runs correctly there. This is one confirmed machine/GPU/toolchain combination, not a claim about every machine — build and run `pytest tests/test_fused_bias_gelu.py -v` on yours to check. **Run the FULL suite together too** (`pytest tests/`), not only this file in isolation — a real bug (fixed; see "1 more bug found" below) only showed up when the full suite ran as a whole, because it depended on which OTHER test file happened to import this module first.

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

## New in 0.8.2: one-call `optimize()`, gradient accumulation, AMD/Intel + NUMA CPU pinning, fast checkpoint loading, per-axis fp8, experimental fp4, fp16 overflow safety, training phase/ETA tracking, pinned-buffer pool, concurrent cache warming — plus 4 confirmed bug fixes

Everything below landed after 0.7.6 and is the current, accurate state of
the package — if anything elsewhere in this README or in
`API_REFERENCE.md` seems to contradict this section, this section (and
the code) wins; open an issue. Verified against the full test suite:
248 tests passing (113 of them new in this release), 0 regressions. (A
follow-up audit pass found 5 more issues in these same additions and
brought the total to 257 passing — see "5 more bugs found and fixed"
below.)

### One call to apply the real perf knobs: `WinCore.optimize()`

```python
import WinCore

plan = WinCore.optimize()   # cpu.apply() then precision.cuda_perf_defaults(), in that order
print(plan.cpu)             # ThreadPlan(...)
print(plan.cuda)            # CudaPerfPlan(...) -- or None if apply_cuda=False
if plan.warnings:
    print(plan.warnings)    # flattened + prefixed ("cpu: ...", "cuda: ...")
```

Call this as the very first WinCore-related thing your script does —
ideally before your own code imports torch for any other reason. This
is a thin sequencing layer, not a new technique: every real effect
already exists in `WinCore.cpu`/`WinCore.precision` with its own tests
and its own docstring; `optimize()` just calls them in the one order
that actually matters (see `WinCore.cpu.apply`'s docstring for exactly
why CPU thread-pool env vars have to be set before torch initializes).

### Real GPU perf tuning in one call: `precision.cuda_perf_defaults()`

```python
plan = WinCore.precision.cuda_perf_defaults()
print(plan.reason)   # e.g. "TF32 enabled (compute capability 8.6). cuDNN benchmark enabled. SDPA backends enabled: ['flash', 'mem_efficient', 'math']."
```

Applies the standard, PyTorch-documented performance knobs
(`torch.backends.cuda.matmul.allow_tf32`, `torch.backends.cudnn.allow_tf32`,
`torch.set_float32_matmul_precision`, `torch.backends.cudnn.benchmark`,
and all three `scaled_dot_product_attention` backend toggles) instead
of every script re-copying the same lines. Checks actual GPU compute
capability first — TF32 is only enabled (and only reported as enabled)
on Ampere+ (>= 8.0); older hardware gets an explanatory warning
instead of a flag that's silently a no-op there. `cudnn_benchmark=False`
if your input shapes vary step to step (cuDNN's shape-keyed autotune
cache works against you there, not for you). `apply=False` returns the
plan without touching anything, for logging/dry-run use.

### Gradient accumulation, done correctly: `WinCore.accumulate.GradientAccumulator`

```python
accum = WinCore.accumulate.GradientAccumulator(accumulation_steps=4, model=ddp_model)
for micro_batch in micro_batches:
    with accum.sync_context():                    # no_sync() on non-boundary steps under DDP
        loss = model(micro_batch)
        accum.scale_loss(loss).backward()          # correct LR -- divides by accumulation_steps
    if accum.step():                                # True only on the boundary micro-step
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
```

Two easy-to-get-silently-wrong details, fixed: forgetting to scale the
loss trains at an effectively N-times-too-high learning rate with **no
error, no crash** — training just "runs" wrong. And under DDP, every
`.backward()` triggers a full gradient all-reduce by default; without
`no_sync()` an N-step accumulation window pays for N all-reduces
instead of one. `sync_context()` is duck-typed against `model.no_sync()`
— works with any object exposing that method, and is a harmless no-op
without DDP.

### AMD/Intel detection + NUMA-aware CPU pinning

```python
print(WinCore.cpu.cpu_vendor())          # "amd" | "intel" | "unknown"
print(WinCore.cpu.numa_node_count())     # 1 on a single-socket machine

plan = WinCore.cpu.apply(affinity=True, numa_aware=True)   # numa_aware=True is the default
print(plan.vendor)
```

On a multi-socket Intel Xeon or AMD Threadripper/EPYC-class machine,
`numa_aware=True` (default) restricts affinity pinning to NUMA node
0's CPUs *before* P-core selection, so a pin doesn't scatter threads
across a NUMA boundary (real, measurable remote-memory latency) as a
side effect of naively taking "the first N logical CPU indices". Falls
back cleanly to the pre-existing behavior if node info can't be
determined or would leave zero usable CPUs.

### Fast checkpoint loading: `WinCore.io.fast_torch_load` / `fast_safetensors_load` / `load_checkpoint`

```python
state = WinCore.io.load_checkpoint("checkpoint.pt", device="cuda:0")
# or explicitly:
state = WinCore.io.fast_torch_load("checkpoint.pt", device="cuda:0", mmap=True)
tensors = WinCore.io.fast_safetensors_load("weights.safetensors", device="cuda:0")
```

`fast_torch_load` defaults to `torch.load(..., mmap=True)` (torch>=2.1)
instead of the classic full-buffer-then-unpickle path, avoiding the
double host-RAM copy a plain `torch.load` pays and letting the OS page
cache do the work. Falls back transparently — never raises — on an
older torch build or a filesystem that rejects mmap (e.g. some network
shares). `load_checkpoint` dispatches by file extension so resume code
doesn't need its own `.safetensors` vs `.pt` if/else.

### Deeper fp8, honest experimental fp4, and fp16 overflow safety

```python
# per-channel fp8 -- one scale per row instead of one for the whole tensor
packed = WinCore.precision.quantize_fp8(weight_matrix, fmt="e4m3", axis=0)
restored = WinCore.precision.dequantize_fp8(packed)   # same call whether axis was given or not

# EXPERIMENTAL: software-only 4-bit storage compression, ~8x smaller than fp32
# -- NOT hardware-accelerated, NOT the recommended default; see its docstring
packed4 = WinCore.precision.quantize_fp4(optimizer_state_tensor)
restored4 = WinCore.precision.dequantize_fp4(packed4)

# fp16's silent overflow-to-inf, made explicit
safe = WinCore.precision.safe_cast(activations, torch.float16, on_overflow="clip")
```

`axis=` on `quantize_fp8` computes an independent scale per position
along that axis instead of one global scale — real precision
improvement when different rows/channels have genuinely different
magnitude distributions. `quantize_fp4`/`dequantize_fp4` are loudly
documented as experimental (see the module for the full list of why:
no tensor-core acceleration, only 15 levels, linear not NF4/AF4) —
they exist for the exploration this package's own docstring already
promised for 4-bit, not to replace a dedicated quantization library.
`safe_cast` catches the specific real failure mode plain float16 has:
a value past ~65504 silently becomes `inf` with **no error, no
warning** on a normal `.to(torch.float16)` call.

### Training phase / progress tracking: `TrainingMonitor.record_step_time()`

```python
monitor = WinCore.diagnostics.TrainingMonitor(expected_param_count=count_params(model))
for step, batch in enumerate(loader):
    status = monitor.record_step_time(step, total_steps=total_steps)
    train_step(batch)
    if status.phase == "stalled":
        print(status.reason)
    if step % 100 == 0 and status.eta_seconds is not None:
        print(f"~{status.eta_seconds/60:.1f} min remaining at current rate")
```

Classifies each step as `warmup` / `steady_state` / `stalled`.
`warmup_steps` defaults to a heuristic scaled by `expected_param_count`
(bigger models get a longer default warmup window, since cuDNN
benchmark autotuning and `torch.compile` first-compile both genuinely
take longer to settle on a larger model) — explicitly an untuned
heuristic, overridable via `warmup_steps=`. ETA is computed only from
steady-state timing, so one slow first step or one stall doesn't skew
it. A `stalled` step also emits a `step_stall` warning `Issue`, same as
this class's other `record_*` methods.

### Pinned-buffer reuse + concurrent cache warming

```python
pool = WinCore.memory.PinnedBufferPool()
buf = pool.get(batch.shape, batch.dtype)
buf.copy_(batch)
# ... transfer buf to GPU ...
pool.release(buf)   # available for the next same-shape get(), no re-pinning

cache = WinCore.cache.DiskCache("D:/wincore_cache", max_bytes=20 * 1024**3)
report = cache.warm(all_dataset_keys, compute_fn=preprocess_one, max_workers=8)
print(f"{report.computed} computed, {report.already_cached} skipped, {report.failed} failed")
```

`PinnedBufferPool` reuses pinned (page-locked) CPU staging buffers by
`(shape, dtype)` instead of re-pinning a fresh buffer every step —
pinning is a real CUDA driver call (`cudaHostAlloc`) with measurable
per-call overhead. `DiskCache.warm()` prepopulates many entries
concurrently via a thread pool ahead of the training loop needing them
— real speedup mostly from overlapping the GIL-releasing C-extension
work most `compute_fn`s spend their time in (image decode/resize,
NumPy, tokenizers), not just SSD write parallelism alone.

### 4 confirmed bugs fixed (silent-wrong-behavior class, not crashes)

- **`import WinCore` alone silently broke `WinCore.cpu.apply()`'s env-var contract.** `WinCore.kernels` eagerly imported torch at package-import time, so torch was already initialized before `apply()` ever got a chance to set `OMP_NUM_THREADS`/`MKL_NUM_THREADS`. Fixed via PEP 562 lazy module `__getattr__` — `import WinCore` no longer touches torch at all until you actually reach into `WinCore.kernels.fused_bias_gelu` or a sibling name.
- **`TrainingMonitor.record_loss()` reported "barely moved... likely stuck" for a loss that was actively diverging**, not stuck — the plateau check didn't distinguish a near-zero relative change from a large *negative* one. Split into `loss_plateau` and a new `loss_regressing`, each with its own message and its own suggested fix (LR too low vs. LR too high).
- **`precision.quantize_fp8()` used the wrong fp8 ceiling for every spelling of "e4m3" except the exact literal string `"e4m3"`** — `fmt="fp8_e4m3"` or `"float8_e4m3fn"` quantized to the correct dtype but scaled against e5m2's ceiling (off by ~128x), silently producing a saturated reconstruction with no warning. Fixed by deriving the ceiling from the *resolved* dtype instead of the raw input string.
- **`cache.DiskCache._evict_if_needed()` could silently drop a still-on-disk file from its own index** when `unlink()` failed (e.g. antivirus/indexer holding the file) — the position counter used to rebuild the index advanced regardless of the failure. Fixed by tracking evicted filenames explicitly.

### 5 more bugs found and fixed in a post-release audit of the 0.8.2 additions themselves

A follow-up, deeper pass specifically targeting the 10 capabilities added earlier in this same 0.8.2 release (stress-testing real concurrency, edge-case inputs, and full cross-module integration) turned up 5 more real issues — version stays 0.8.2 (this is a same-version fix pass, not a new release):

- **`DiskCache.warm()` called a full directory rescan under a cross-process file lock after EVERY SINGLE key**, not once per `warm()` call — for K keys, K serialized lock acquisitions and full-directory rescans, actively working against the concurrency `warm()` exists to provide. There was also a dead line of code: the original intent (evict once, at the end) was written *after* the function's `return` statement, so it was syntactically valid but unreachable and never ran. Both fixed together — see `warm()`'s own docs above for the corrected behavior.
- **`quantize_fp8()`/`quantize_fp4()` crashed with a confusing raw numpy error on an empty (0-element) tensor** instead of a clear message. Both now raise an actionable `ValueError`.
- **`quantize_fp8(tensor, axis=<out of range>)` silently accepted an invalid axis** instead of raising — e.g. `axis=5` on a 2D tensor didn't match any real dimension, so it silently reduced over the whole tensor (identical to `axis=None`) while still labeling the result with the bogus axis value. Now validates the axis is in range and raises `ValueError` otherwise; valid negative indices (`axis=-1`, etc.) are unaffected.
- **`GradientAccumulator(accumulation_steps=2.5)` (or any non-integer) was silently accepted**, producing an inconsistent accumulation-window size (e.g. `2.5` alternates between 3-step and 2-step windows) that quietly breaks the fixed-effective-batch-size guarantee this class exists to provide. Now raises `TypeError` for anything that isn't a plain `int`.

Full test suite: 257 passed (up from 248 — 9 new regression tests added alongside these fixes), 0 regressions.

### 2 more bugs found on real Windows + CUDA hardware, via a real full-suite `pytest tests/ -v` run

Everything above this point had only ever been run in a sandbox with no GPU, against a fake-torch test shim standing in for real tensor operations. Running the actual full suite on real Windows 11 + CUDA + MSVC hardware surfaced two real issues that sandbox testing alone couldn't catch — version stays 0.8.2:

- **`WinCore.kernels.fused_bias_gelu` intermittently resolved to the submodule (a module object) instead of the function**, `TypeError: 'module' object is not callable` — but only when running the FULL suite (`pytest tests/`); `pytest tests/test_fused_bias_gelu.py` alone always passed, which is exactly what made this easy to miss. Root cause: the submodule implementing this kernel was named `fused_bias_gelu.py` — identical to the public function name it exports — and Python's import system unavoidably binds an imported submodule onto its parent package's namespace under the submodule's own name as a side effect of ANY import of it, not only ones going through this package's own lazy-loading. A different test file needed a private helper not exposed through the lazy public API and imported the submodule directly, which silently poisoned the public `fused_bias_gelu` attribute for every test that ran afterward. Fixed at the root: the file is renamed to `_fused_bias_gelu_kernel.py` so the collision can't happen again, regardless of import order or what imports it directly. The public `WinCore.kernels.fused_bias_gelu` attribute is completely unchanged.
- **Two `quantize_fp8()` tests asserted a near-exact round-trip that only ever passed against the sandbox's fake torch shim**, which doesn't perform real bit-level fp8 rounding. On real hardware, e4m3's ~3 mantissa bits produce a genuine several-percent relative error per element — fp8 doing exactly what fp8 is supposed to do, not a library bug. The actual `quantize_fp8`/`dequantize_fp8` code was never wrong; only two test assertions encoded an unrealistic "lossless" expectation. Fixed by loosening both to the same 15% tolerance the pre-existing, already-hardware-validated fp8 test in `test_precision.py` already uses.

Full test suite: still 257 passed in the sandbox, 0 regressions from these fixes.



Everything below landed after 0.7.1 and is the current, accurate state of
the package — if anything elsewhere in this README or in
`API_REFERENCE.md` seems to contradict this section, this section (and
the code) wins; open an issue.

**`WinCore.cache.DiskCache`'s `max_bytes` budget is now enforced across
every process sharing the cache directory**, not just the process that
created a given instance. This matters specifically because
`DataLoader(num_workers>0)` always uses `spawn` on Windows (never `fork`,
unlike Linux) — each worker gets its own independent `DiskCache` object
with independent local bookkeeping, so without cross-process enforcement
N workers could each "correctly" stay under `max_bytes` on their own
while the real directory grew toward roughly `N * max_bytes`. Fixed with
a dependency-free cross-process lock file and real directory rescans
instead of trusting any one process's local state — automatic, no code
changes needed on your end. See `WinCore.cache` in `API_REFERENCE.md`
for the full mechanism.

**`WinCore.cpu.pin_affinity()` (and `apply(affinity=True)`, which is what
the Quick Start below actually uses) now detects real performance-core
(P-core) logical CPUs on Intel hybrid (12th gen+) CPUs** via
`GetLogicalProcessorInformationEx`, instead of assuming the first N
logical CPU indices are the fast ones — an assumption that could
silently pin a process onto efficiency cores on some real hybrid
layouts, the opposite of what affinity pinning is for. Both
`pin_affinity()` called directly and `apply(affinity=True)` share the
exact same P-core-aware selection now (an earlier internal version had
`apply()` bypass it — fixed, with an integration-level regression test
guarding against that specific class of bug recurring).

**`WinCore.kernels.build()`'s MSVC auto-detection now finds Visual Studio
installs on the Preview/Insider channel**, not just stable releases —
confirmed on a real Visual Studio 2026 (MSVC v145) machine where the
previous `vswhere` query returned nothing at all despite the C++ Build
Tools being genuinely installed, silently falling back to an unfused
kernel with misleading "Build Tools not installed" guidance.

**New module, `WinCore.power`** — two Windows-specific gaps for
unattended AI training with no Linux equivalent by default:
- `prevent_sleep()`: stops Windows from suspending the machine mid-run
  (a training loop produces none of the user-input activity Windows'
  idle timer looks for), via the real `SetThreadExecutionState` API.
- `check_tdr_risk()`: reads Windows' GPU driver watchdog timeout
  (`TdrDelay`) and explains whether a long single CUDA kernel launch —
  exactly the kind `WinCore.kernels` ships — risks being killed and
  surfacing as a `CUDA error: unspecified launch failure` with no
  indication a 2-second Windows timer, not a bug, caused it.

```python
with WinCore.power.prevent_sleep():
    for epoch in range(num_epochs):
        train_one_epoch(...)
```

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

# 1. One call for the standard CPU + CUDA perf knobs (0.8.2+). Call this
# FIRST -- before your own code imports torch for any other reason, since
# the CPU thread-pool env vars only take effect if set before torch inits.
plan = WinCore.optimize()
print(plan.cpu, plan.cuda)
if plan.warnings:
    print("non-fatal warnings:", plan.warnings)

# Equivalent to (and exactly what optimize() calls, in this order, if you
# want the individual pieces / more control over each):
#   WinCore.cpu.apply(priority="above_normal", affinity=True, numa_aware=True)
#   WinCore.precision.cuda_perf_defaults()

# 2. Check the machine actually meets your training script's minimums
check = WinCore.spec.meets_minimum(min_vram_gb=6, min_ram_gb=16)
if not check.ok:
    raise SystemExit(f"Machine doesn't meet requirements: {check.reasons}")

# 3. Pick a safe default dtype for this GPU
dtype = WinCore.precision.recommended_dtype()

# 4. Compile safely (falls back to eager if Triton misbehaves on Windows)
model = WinCore.safe_compile(model)

# 5. Save checkpoints without random WinError 32 crashes, load them fast
WinCore.atomic_write(lambda p: torch.save(model.state_dict(), p), "checkpoint.pt")
state = WinCore.io.load_checkpoint("checkpoint.pt", device="cuda:0")

# 6. Don't let Windows suspend the machine mid-run, and know up front
# whether a long custom-kernel launch risks getting killed by the
# driver watchdog (see "New in 0.7.1" below for why this matters):
tdr = WinCore.power.check_tdr_risk()
if tdr.at_default_risk_level:
    print(tdr.message)

with WinCore.power.prevent_sleep():
    for epoch in range(num_epochs):
        train_one_epoch(...)
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

### Gradient accumulation (larger effective batch size than fits in VRAM)

```python
import WinCore

accum = WinCore.accumulate.GradientAccumulator(accumulation_steps=4)  # add model=ddp_model for DDP no_sync() skipping
for micro_batch in micro_batches:
    with accum.sync_context():
        loss = model(micro_batch)
        accum.scale_loss(loss).backward()
    if accum.step():
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
```

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

# optional: prepopulate the cache concurrently before training starts,
# instead of paying every miss serially during epoch 1
cache.warm(range(len(dataset)), compute_fn=lambda idx: preprocess(dataset.raw(idx)), max_workers=8)

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
- **Not a native fp4 compute type.** There's still no IEEE/CUDA-native 4-bit compute dtype — `precision.resolve_dtype("fp4")` raises `ValueError` on purpose rather than pretending to support it as a *compute* dtype. `precision.quantize_fp4()`/`dequantize_fp4()` (0.8.2+) exist as an explicitly EXPERIMENTAL, non-hardware-accelerated *storage* compression scheme (a plain-Python bit-packing loop, not a vectorized/fast path) for exploratory use — not a bitsandbytes/AWQ/GPTQ replacement for production 4-bit weight quantization. See that function's own docstring for the full list of caveats before reaching for it.
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
