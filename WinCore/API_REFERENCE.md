# WinCore API Reference

Every public function and class in WinCore, what it takes, what it returns, and what it actually does. `README.md` is the pitch and quick-start; this is the reference you check when you need to know a specific parameter or behavior. Nothing below is invented — it's pulled directly from the docstrings and logic in the source.

Anything starting with `_` (e.g. `_default_reserve`, `_ram_info`) is a private helper, not part of the public API, and isn't listed here.

---

## `WinCore` (top level)

### `WinCore.docs() -> str`
Returns the full text of `README.md`, which ships inside the installed package specifically so it survives `pip install` (a plain `readme = "README.md"` entry in `pyproject.toml` only embeds it into the PyPI page metadata — it does not, by itself, leave a browsable file in `site-packages`). Falls back to a pointer message if the README file is somehow missing from this build.

```python
import WinCore
print(WinCore.docs())
```

Also re-exported directly: `atomic_write`, `AtomicWriteError`, `safe_compile`, `should_compile`, `SafeCompiled` — everything else lives under its module (`WinCore.spec`, `WinCore.cpu`, etc.), shown below.

`help(WinCore)` gives a shorter, module-by-module overview instead of the full README.

---

## `WinCore.io`

### `atomic_write(write_fn, dst, retries=6, initial_delay=0.25, max_delay=3.0) -> None`
Writes a file atomically — write to a temp file, then `os.replace()` it into place — with retry/backoff specifically because Windows raises `WinError 32` ("file in use by another process") far more readily than Linux when something else (antivirus, indexer, a backup tool) briefly holds a lock on the destination.

| param | meaning |
|---|---|
| `write_fn` | called once with a temp `Path`; must fully write **and close** the file itself (e.g. inside a `with open(p, "w") as f:` block) |
| `dst` | final destination path; its parent directory must already exist |
| `retries` | max attempts for the final `os.replace()` step |
| `initial_delay` | seconds before the first retry |
| `max_delay` | cap on exponential backoff between retries |

Raises `AtomicWriteError` if `os.replace()` still fails after all retries (the destination is left untouched, only the temp file is lost). If `write_fn` itself raises, that exception propagates directly and the destination is never touched either way.

```python
WinCore.atomic_write(lambda p: torch.save(model.state_dict(), p), "checkpoint.pt")
```

### `AtomicWriteError(OSError)`
Raised when a write couldn't be committed after all retries. Subclasses `OSError`, so existing `except OSError:` handling still catches it.

### `atomic_torch_save(obj, dst, retries=6, initial_delay=0.25, max_delay=3.0, **torch_save_kwargs) -> None`
Convenience wrapper: same atomicity/retry guarantee as `atomic_write`, saving the `lambda p: torch.save(obj, p)` boilerplate. `obj` is anything `torch.save` accepts. `**torch_save_kwargs` forwards straight through (e.g. `pickle_protocol=...`). `torch` is imported lazily inside the function, so `WinCore.io` itself keeps zero hard dependencies.

```python
WinCore.atomic_torch_save(model.state_dict(), "checkpoint.pt")
```

### `atomic_safetensors_save(tensors, dst, metadata=None, retries=6, initial_delay=0.25, max_delay=3.0) -> None`
Same atomicity guarantee, via `safetensors.torch.save_file()`. `tensors` must be a flat `dict[str, torch.Tensor]` (safetensors' own requirement — it can't serialize arbitrary nested Python objects, which is exactly what makes it a safer load target than pickle-based formats). `metadata`, if given, must be `dict[str, str]`. Needs the optional `safetensors` package (`pip install safetensors`, or the `WinCore[safetensors]` extra) — raises a clear `ImportError` naming the install command if it's missing, not a confusing `AttributeError` from deep in some other code path.

```python
WinCore.atomic_safetensors_save(model.state_dict(), "checkpoint.safetensors")
```

---

## `WinCore.compile`

### `should_compile() -> bool`
Whether `torch.compile` should be attempted by default here. Resolution order:
1. env var `WINML_SAFE_DISABLE_COMPILE=1` → always `False`
2. env var `WINML_SAFE_FORCE_COMPILE=1` → always `True` (Triton's Windows support is unofficial — this is an explicit opt-in override, not a recommendation)
3. otherwise: `False` on Windows, `True` elsewhere

### `safe_compile(module, mode=None, fullgraph=False, enabled=None, on_fallback=None, **compile_kwargs) -> Any`
Wraps `module` with `torch.compile`, with automatic eager fallback built in.

| param | meaning |
|---|---|
| `module` | an `nn.Module` (or any callable) to compile |
| `mode` | passed straight through to `torch.compile` (e.g. `"reduce-overhead"`) |
| `fullgraph` | passed straight through to `torch.compile` |
| `enabled` | override `should_compile()` just for this call — `False` forces eager, `True` forces a compile attempt regardless of platform |
| `on_fallback` | called once, the first time the compiled path fails **at runtime** — receives the exception, so you can log it instead of it failing silently |
| `**compile_kwargs` | forwarded to `torch.compile()` |

Returns either `module` unchanged (compilation disabled, or `torch.compile` unavailable/failed to wrap at all), or a `SafeCompiled` proxy that behaves like the compiled callable but permanently falls back to eager the first time it raises at call time.

```python
model = WinCore.safe_compile(model)   # compiles where safe, silently eager on Windows by default
```

### `SafeCompiled`
Callable proxy returned by `safe_compile`. `SafeCompiled(eager_module, compiled_callable, on_fallback=None)`. Call it like the original module (`SafeCompiled(x)`); `.fell_back` (a **property**, not a method — access as `compiled.fell_back`, no parentheses) reports whether it has already dropped to eager mode. Any attribute not defined on `SafeCompiled` itself (`.parameters()`, `.state_dict()`, `.to()`, `.train()`, `.eval()`, custom attributes, ...) is transparently forwarded to the underlying eager module via `__getattr__`, so `SafeCompiled` is a real drop-in replacement for the wrapped `nn.Module` in a training loop, not just for its forward call. This delegates to the eager module specifically (not the compiled callable) — `torch.compile()` optimizes the forward pass, it does not clone the module or its parameters, so `.parameters()`/`.state_dict()` return the same, consistent result whether the compiled or eager path is currently active.

---

## `WinCore.cpu`

Tiered logical-thread reservation: by default this leaves a small slice of threads free for the OS/antivirus/indexer instead of pinning 100% of them, since claiming every thread tends to run *slower* and less stably on a real desktop, not faster.

| total logical threads | reserved | used by default |
|---|---|---|
| ≤ 8 | 1 | total − 1 |
| 9–12 | 2 | total − 2 |
| 13–16 | 3 | total − 3 |
| > 16 | up to 4 (scales down) | total − reserve |

### `recommended_threads(total=None, reserve=None, threads=None) -> ThreadPlan`
Computes the plan without touching the process.

| param | meaning |
|---|---|
| `total` | override the detected thread count (mainly for testing); defaults to `os.cpu_count()` |
| `reserve` | force this many threads reserved, instead of the tiered default above |
| `threads` | skip the heuristic entirely, use exactly this many (clamped to `[1, total]`) |

### `apply(total=None, reserve=None, threads=None, set_env=True, priority=None, affinity=False, strict=False) -> ThreadPlan`
Computes the plan (same params as above) **and applies it**, best-effort:
- `torch.set_num_threads(...)` if torch is importable
- `OMP_NUM_THREADS` / `MKL_NUM_THREADS` env vars if `set_env=True` — but only takes effect on NumPy/MKL/OpenMP libraries if set *before* those libraries were imported; setting them afterward has no effect on the current process (a Python-level limitation, not a bug here)
- if `priority` is set (e.g. `"above_normal"`), calls `set_priority(priority)`
- if `affinity=True`, calls `pin_affinity(range(plan.recommended))`

The `priority`/`affinity` calls are real OS-scheduler-level requests (see `set_priority`/`pin_affinity` below) — they reach the Python main loop and DataLoader worker processes, which thread-pool sizing alone cannot. Failures there (missing `psutil`, OS denial) are non-fatal by default and land in `plan.warnings`; pass `strict=True` to raise `PriorityError` instead.

Returns the applied `ThreadPlan` so you can log it.

```python
plan = WinCore.cpu.apply(priority="above_normal", affinity=True)
print(plan)            # ThreadPlan(total_logical=12, reserved=2, recommended=10)
print(plan.warnings)   # () if everything applied cleanly
```

### `set_priority(level="above_normal", pid=None) -> str`
Sets real OS scheduling priority (`SetPriorityClass` on Windows, `nice`/`setpriority` on POSIX, via `psutil`). `level` is one of `"idle"`, `"below_normal"`, `"normal"`, `"above_normal"`, `"high"`. Pass `pid` to adjust a different process (e.g. a DataLoader worker). Raises `PriorityError` if `psutil` is missing, the level name is invalid, or the OS denies the change.

### `pin_affinity(cpus=None, pid=None) -> list`
Pins the process to specific logical CPUs (`SetProcessAffinityMask` / `sched_setaffinity`, via `psutil`). `cpus` defaults to `range(recommended_threads().recommended)`. Not supported on macOS (no public affinity API there) — raises `PriorityError` in that case, same as any other OS/permission denial.

### `PriorityError(RuntimeError)`
Raised by `set_priority`/`pin_affinity` (and re-raised by `apply(..., strict=True)`) on missing `psutil`, an invalid level, an unsupported platform, or an OS/permission denial.

### `ThreadPlan`
Frozen dataclass: `total_logical: int`, `reserved: int`, `recommended: int`, `warnings: tuple = ()`.

---

## `WinCore.spec`

Real hardware readout — no invented numbers. Everything here reads what the OS/driver actually reports, and returns `None`/empty when it genuinely can't be measured, rather than guessing.

### `get_system_spec() -> SystemSpec`
Full snapshot: OS name, logical/physical CPU counts, RAM total/available (GB), and every detected GPU. Needs `psutil` for CPU/RAM detail and `pynvml`/torch/`wmi` for GPU detail (all optional — degrades to `None`/empty fields, not an exception, if a dependency is missing).

### `get_gpus() -> List[GPUInfo]`
Tries, in order: `pynvml` (best VRAM detail, NVIDIA) → `torch.cuda` (if torch installed) → Windows WMI (name only, any vendor). Empty list if none apply — not an error.

### `get_gpu_temperature(index=0) -> Optional[float]`
One direct `pynvml.nvmlDeviceGetTemperature` call (NVIDIA's own helper, not a custom sensor) — cheaper than a full `get_system_spec()` when polling every N steps. Returns `None` on any failure (no pynvml, no NVIDIA GPU, driver doesn't expose it); never raises, so a training-loop poll can't crash the run.

### `meets_minimum(min_logical_threads=1, min_vram_gb=None, min_ram_gb=None, cpu_name_contains=None, spec=None) -> MinimumCheckResult`
Checks the machine against caller-supplied minimums — only for things that can actually be measured portably.

| param | meaning |
|---|---|
| `min_logical_threads` | fails if `spec.logical_threads` is below this |
| `min_vram_gb` | fails if no GPU, or best detected GPU's total VRAM is below this |
| `min_ram_gb` | fails if RAM total is below this (or unknown, if `psutil` isn't installed) |
| `cpu_name_contains` | optional soft allow-list of substrings (e.g. `["i5-9", "ryzen 5"]`) checked against `platform.processor()` — **advisory only**, since that string isn't standardized across vendors/OSes; there's no reliable, portable way to detect CPU *generation* from Python |
| `spec` | pass an already-computed `SystemSpec` to skip re-detecting |

Returns `MinimumCheckResult(ok: bool, reasons: List[str], spec: SystemSpec)` — `reasons` lists exactly what failed and why, including "unknown" cases (e.g. `"RAM size unknown (install 'psutil' to check this)"`).

```python
check = WinCore.spec.meets_minimum(min_vram_gb=6, min_ram_gb=16)
if not check.ok:
    raise SystemExit(f"Machine doesn't meet requirements: {check.reasons}")
```

### `GPUInfo`
`name`, `vendor` (`"nvidia"|"amd"|"unknown"`), `vram_total_gb`, `vram_free_gb`, `temperature_c`, `source` (which backend supplied this entry) — all `Optional`.

### `SystemSpec`
`os_name`, `logical_threads`, `physical_cores`, `ram_total_gb`, `ram_available_gb`, `gpus: List[GPUInfo]`.

> **On `ram_available_gb`:** this is `psutil.virtual_memory().available` — the same number Windows Task Manager shows as "Available" (RAM genuinely free for new allocations without swapping), **not** `total − used`. Other running programs (browser tabs, background apps) legitimately reduce this number; a low reading isn't necessarily a WinCore bug — compare against Task Manager's own "Available" figure at the same moment to check.

### `MinimumCheckResult`
`ok: bool`, `reasons: List[str]`, `spec: SystemSpec`.

---

## `WinCore.precision`

### `recommended_dtype(device_index=0)`
Picks a safe default dtype from the GPU's actual compute capability (requires torch):
- no CUDA → `torch.float32`
- compute capability ≥ 8.0 (Ampere+) → `torch.bfloat16`
- compute capability ≥ 6.0 (Pascal+, e.g. GTX 10-series) → `torch.float16`
- older → `torch.float32` (fp16 tensor cores aren't present; emulated fp16 is often slower, not faster)

### `resolve_dtype(name: str)`
Resolves a string to a torch dtype: `"fp16"`, `"bf16"`, `"fp32"`, `"fp64"`, `"fp8"`/`"fp8_e4m3"`/`"fp8_e5m2"`, or full torch names. fp8 needs PyTorch 2.1+ (raises a clear `ValueError`, not a confusing `AttributeError`, on older torch). **`"fp4"` intentionally raises `ValueError`** — 4-bit is a quantization scheme (e.g. via `bitsandbytes`), not a compute dtype, and this function won't silently pretend otherwise.

### `dtype_name(dtype) -> str`
Inverse of `resolve_dtype` — short name for a torch dtype.

### `amp(device_index=0) -> AmpContext`
Bundles `torch.autocast` + `torch.amp.GradScaler` around the dtype from `recommended_dtype()`, so mixed precision is one call instead of separately picking a dtype, remembering `GradScaler` only matters for `float16` (not `bfloat16` — no loss-scaling need there), and handling the no-CUDA case by hand.

Returns an `AmpContext`:
- `.autocast()` — `torch.autocast` context manager, already configured with the right device type/dtype; a real no-op (`enabled=False`) on CPU-only or pre-fp16-tensor-core GPUs.
- `.scaler` — `torch.amp.GradScaler`, enabled only for `float16`.
- `.plan` — an `AmpPlan(enabled, dtype, use_grad_scaler, reason)` explaining the resolution.

```python
ctx = WinCore.precision.amp()
for x, y in loader:
    optimizer.zero_grad()
    with ctx.autocast():
        loss = model(x, y)
    ctx.scaler.scale(loss).backward()
    ctx.scaler.step(optimizer)
    ctx.scaler.update()
```

Does not implement mixed precision itself — the numerics are entirely `torch.autocast`/`GradScaler`'s; this only wires the dtype decision into them consistently.

### `quantize_fp8(tensor, fmt="e4m3") -> Fp8Tensor`
Compresses a float tensor to real fp8 **storage** (as opposed to `amp()`'s compute-dtype selection) with dynamic per-tensor scaling, so the tensor's own value range is fit into fp8's narrow representable range instead of a fixed scale that's wrong for most tensors. `fmt` is `"e4m3"` (default; more mantissa, better for weights/activations) or `"e5m2"` (more exponent range, better for gradients). Needs a Hopper/Ada+ GPU (compute capability ≥ 8.9) and torch 2.1+ — same requirement, same clear `ValueError` on mismatch, as `resolve_dtype("fp8")`.

### `dequantize_fp8(packed: Fp8Tensor)`
Inverse of `quantize_fp8` — restores an approximation at the original dtype. Lossy by design (fp8's mantissa is too narrow to round-trip exactly); returns the best reconstruction available from the kept scale + fp8 data.

### `Fp8Tensor`
Dataclass: `data` (the fp8 tensor), `scale: float`, `orig_dtype`.

---

## `WinCore.thermal`

### `ThermalGuard(threshold_c=83.0, pause_seconds=5.0, gpu_index=0, on_pause=None, monitor=None, critical_threshold_c=None, on_critical=None, max_pause_seconds=30.0, backoff_factor=1.5)`
Call `.check(step=None)` periodically (e.g. every N steps). Reads temperature via `pynvml.nvmlDeviceGetTemperature` (NVIDIA's own call, not a custom sensor) and sleeps if over `threshold_c`, giving existing cooling time to catch up — this is monitoring plus a software-level pause, **not** hardware fan/clock/power control (Python has no portable access to that layer; the driver/BIOS/vendor overlay already throttles independently of this).

**Graduated pause, not fixed:** the sleep duration escalates geometrically (`pause_seconds * backoff_factor ** consecutive_overheat_checks`, capped at `max_pause_seconds`) the longer temperature has stayed above `threshold_c` on consecutive checks, and resets to the base `pause_seconds` the moment a check comes back under threshold — a brief spike gets a short pause, sustained overheating gets progressively more aggressive cooldown, instead of the same fixed-duration pause every time regardless of severity.

**`critical_threshold_c`** (optional, set above `threshold_c`) marks a more severe line — if a reading meets or exceeds it, `on_critical(event)` fires **before** the sleep, specifically so a caller can save a checkpoint (e.g. via `WinCore.io.atomic_torch_save`) while there's still time. This module never decides to abort training on your behalf — `on_critical` is a hook for you to act on, not a built-in kill switch.

`.check()` returns a `ThermalEvent` if it paused, else `None`. If temperature can't be read at all (no pynvml, no NVIDIA GPU), it silently returns `None` — this guard degrades to a no-op rather than breaking a loop that doesn't have GPU monitoring available.

If constructed with `monitor=some_training_monitor`, every reading (not just pauses) is fed into `monitor.record_signal(step, "gpu_temp_c", temp)` automatically, so a loss/gradient issue near that step gets annotated with the temperature at the time.

```python
guard = WinCore.thermal.ThermalGuard(threshold_c=83)
for step, batch in enumerate(loader):
    train_step(batch)
    if step % 20 == 0:
        guard.check(step)

# with graduated backoff (default) and a critical auto-save hook:
guard = WinCore.thermal.ThermalGuard(
    threshold_c=83, critical_threshold_c=90,
    on_critical=lambda e: WinCore.io.atomic_torch_save(model.state_dict(), "emergency_checkpoint.pt"),
)
```

### `ThermalEvent`
`temperature_c: float`, `threshold_c: float`, `paused_seconds: float`, `critical: bool = False` (True if this reading also crossed `critical_threshold_c`).

---

## `WinCore.diagnostics`

### `TrainingMonitor(loss_plateau_window=50, loss_plateau_min_relative_improvement=0.001, grad_explode_factor=10.0, grad_vanish_threshold=1e-7, grad_vanish_patience=20, signal_correlation_window=3, on_issue=None)`
Attach to a training loop to catch the "didn't crash, but something's silently wrong" class of problems, and to report where wall-clock time actually goes. Every `record_*` call is cheap and safe to call every step — it only observes, calls `on_issue` if given one, and appends to an inspectable log; it never touches your model/optimizer/data and never raises on your behalf.

Methods:

| method | what it does |
|---|---|
| `record_loss(step, loss_value) -> Optional[Issue]` | NaN/Inf detected immediately; a plateau flagged once `loss_plateau_window` values have accumulated with rolling relative improvement below `loss_plateau_min_relative_improvement` |
| `record_grad_norm(step, model) -> Optional[Issue]` | call **after** `.backward()`, **before** `.step()`/`.zero_grad()`; flags an explosion (norm jumps by `grad_explode_factor`×) or vanishing (stays below `grad_vanish_threshold` for `grad_vanish_patience` steps). Needs torch. |
| `record_signal(step, name, value, note=None) -> None` | feed in any external reading — GPU temp, VRAM pressure, an LR change, anything. Purely observational, never raises an Issue by itself. Any loss/gradient Issue emitted within `signal_correlation_window` steps gets annotated with whichever signals were recorded nearby — a **co-occurrence** note, not a causal claim. |
| `data_timer()` / `compute_timer()` | context managers: `with monitor.data_timer(): batch = next(it)` / `with monitor.compute_timer(): loss = train_step(batch)` |
| `gpu_timer()` | context manager measuring **actual GPU-clock busy time** (`torch.cuda.Event`) for the wrapped block, next to its CPU wall-clock time — catches the gap between "CPU thread is inside the compute block" and "GPU is actually executing", which a coarse utilization-percent reading (`nvidia-smi`) doesn't show. Calls `event.synchronize()` once per use (a real small stall), so it's meant for periodic sampling (e.g. every N steps), not necessarily every step. Degrades to a wall-clock-only no-op if CUDA (or torch) isn't available — never raises for that reason. |
| `bottleneck_report() -> dict` | accumulated data-wait vs. compute time; flags a DataLoader bottleneck if data-wait is ≥40% of total (a real, invisible-otherwise failure mode — loss looks fine, GPU is just idling between batches). If any `gpu_timer()` samples were taken, also includes `gpu_busy_seconds` / `gpu_block_wall_seconds` / `gpu_idle_fraction`, and flags a `gpu_launch_stall` Issue if idle time is ≥20% of the measured GPU-block wall time — identifying *that* the GPU stalled inside that block, not *why*; use `torch.profiler` for the latter. |
| `summary() -> List[Issue]` | every issue recorded so far, in order |

```python
monitor = WinCore.diagnostics.TrainingMonitor(on_issue=lambda i: print(i.severity, i.message))
for step, batch in enumerate(loader):
    with monitor.data_timer():
        x, y = batch
    with monitor.compute_timer():
        loss = train_step(x, y); loss.backward()
    monitor.record_loss(step, loss.item())
    monitor.record_grad_norm(step, model)
    optimizer.step(); optimizer.zero_grad()
monitor.bottleneck_report()
```

### `attach_nan_guards(model, on_issue=None, check_forward=True, check_backward=True, raise_on_detect=False) -> NaNGuardHandle`
Hooks every submodule (`model.named_modules()`, including `model` itself) to flag the **first** module whose forward output or backward input-gradient contains NaN/Inf — pinpointing where the problem originates, not just that it eventually reached the loss.

| param | meaning |
|---|---|
| `on_issue` | called with an `Issue` (`code="module_output_nan"` or `"module_grad_nan"`) the first time a hooked module misbehaves |
| `check_forward` / `check_backward` | which hooks to register |
| `raise_on_detect` | if `True`, raise `FloatingPointError` immediately instead of/alongside calling `on_issue` |

Only the first tensor-shaped output/grad of each module is checked directly, but multi-output modules (e.g. an LSTM returning `(out, (h, c))`) are walked recursively through tuples/lists so every tensor leaf gets checked, not just position 0.

Returns a `NaNGuardHandle` — call `.detach()` to remove every hook (or use it as a context manager: `with attach_nan_guards(model) as guard: ...`). Leaving hooks attached costs a per-module finite-check per forward/backward call — real but usually small next to the actual matmul/conv work.

### `Issue`
`step: Optional[int]`, `code: str`, `severity: str`, `message: str`, `data: dict`.

### `NaNGuardHandle`
`.detach()` removes all hooks; `.enabled = False` pauses detection without re-registering later; usable as a context manager.

---

## `WinCore.multigpu`

### `plan_distributed(gpu_count=None) -> DistributedPlan`
Puts together a full recommendation for the detected (or given) GPU count. **Does not call `init_process_group`** — this only recommends settings; you wire them into your own launch (`torchrun`, `mp.spawn`, etc.).

### `recommended_backend() -> str`
`"nccl"` wherever it's actually available (Linux, incl. WSL2) — `"gloo"` on native Windows, since PyTorch ships without an official NCCL build there. Pure platform detection, no torch import required.

### `detect_topology(gpu_count=None) -> TopologyReport`
**Actually measures** GPU interconnect topology — which pairs are NVLink-connected (and how many links), and for pairs that aren't, PCIe distance (same switch / crosses a host bridge / crosses NUMA nodes / crosses sockets). Tries `pynvml` first (direct NVML calls), then falls back to parsing `nvidia-smi topo -m` output if `pynvml` isn't installed but the CLI is on PATH.

Returns `measured=False` with empty `links` if neither path works (no NVIDIA tooling, AMD GPUs, or fewer than 2 GPUs) — this is never fabricated; `measured` tells the caller whether `recommended_bucket_cap_mb()` is about to use a real reading or a fallback heuristic.

### `recommended_bucket_cap_mb(gpu_count=None, topology=None) -> int`
DDP's own default is 25MB. When topology **is measured**: all-NVLink pairs keep 25 (already amortizes launch overhead well on high-bandwidth links); otherwise the *worst* measured PCIe hop for any pair drives it up (single switch → 40, multiple switches → 60, host bridge → 80, NUMA node → 90, crosses sockets → 100 — a link with a worse hop benefits more from fewer, larger transfers). When topology **can't be measured**, falls back to a GPU-count heuristic (≤1→25, 2→50, 3-4→75, >4→100) — labeled as a heuristic to benchmark, not a measurement.

### `ddp_kwargs(plan=None, find_unused_parameters=False) -> dict`
Kwargs dict ready to splat into `DistributedDataParallel(model, **kwargs)`. `find_unused_parameters` is left as an explicit argument you must decide — this module has no visibility into whether your forward pass conditionally skips parameters, and guessing wrong in either direction has a real cost.

### `check_gpu_balance(imbalance_threshold=1.5) -> GPUBalanceReport`
Reads free VRAM per visible GPU (`torch.cuda.mem_get_info`) and flags if one card has meaningfully less headroom than the others — the common real cause of "training OOMs on rank 2 only, twenty minutes in" (e.g. one card is also driving the display). Empty report (no warning) if CUDA isn't available or fewer than 2 GPUs are visible.

### `detected_gpu_count() -> int`
How many CUDA devices torch currently sees. `0` if torch isn't installed or nothing's visible — never raises.

### `init_from_env(plan=None, timeout_seconds=1800) -> DistributedPlan`
Reads the standard `torchrun`/`mp.spawn` environment variables (`RANK`, `WORLD_SIZE`, `LOCAL_RANK`) and calls `torch.distributed.init_process_group()` with `plan`'s backend (or a freshly computed `plan_distributed()` if `plan` isn't given), then `torch.cuda.set_device(local_rank)` so this process lands on the right GPU. Does **not** replace `torchrun`/`mp.spawn` — something still has to launch the N processes and set those env vars first; this only wires the already-set values into one call instead of the caller re-deriving them. Raises `RuntimeError` with a specific message (not a raw `KeyError`) if the required env vars aren't set. Returns the `DistributedPlan` used, for logging.

```python
plan = WinCore.multigpu.plan_distributed()
plan = WinCore.multigpu.init_from_env(plan)   # torchrun --nproc_per_node=N your_script.py
model = DDP(model, **WinCore.multigpu.ddp_kwargs(plan))
```

### `local_rank_from_env() -> Optional[int]`
Reads `LOCAL_RANK` the way `torchrun` sets it. `None` if not running under a launcher that sets it — a plain env-var read, not a "should I be distributed" decision (that's left to the caller).

```python
plan = WinCore.multigpu.plan_distributed()
topo = WinCore.multigpu.detect_topology()
balance = WinCore.multigpu.check_gpu_balance()
model = DDP(model, **WinCore.multigpu.ddp_kwargs(plan, find_unused_parameters=False))
```

### Dataclasses
`DistributedPlan(backend, world_size, bucket_cap_mb, gradient_as_bucket_view, reason)` · `GPULink(gpu_a, gpu_b, nvlink, nvlink_count, pcie_rank, label)` · `TopologyReport(gpu_count, links, measured, source, all_nvlink, worst_pcie_rank, note)` · `GPUBalanceReport(free_gb, total_gb, imbalance_ratio, warning)`.

---

## `WinCore.memory`

### `recommended_dataloader_kwargs(cpu_recommended_threads=None, cuda_available=None) -> DataLoaderPlan`
Windows-aware `DataLoader(**kwargs)` defaults — accounts for Windows using the `spawn` start method (not `fork`), which re-imports your whole script per worker, so worker count is capped below the raw CPU count instead of matching it 1:1.

| param | meaning |
|---|---|
| `cpu_recommended_threads` | pass `WinCore.cpu.recommended_threads().recommended` if already computed, to avoid detecting CPU count twice; detected fresh via `os.cpu_count()` if omitted |
| `cuda_available` | pass `torch.cuda.is_available()` if already known; detected lazily (best-effort) otherwise, treated as `False` if torch import fails |

### `CacheGuard(min_free_fraction=0.10, gpu_index=0, adaptive=False, trend_window=6, lookahead_checks=3)`
`.check() -> Optional[MemoryPressureEvent]` calls `torch.cuda.empty_cache()` when free VRAM has actually dropped below `min_free_fraction` — not on a fixed schedule, which forces a CUDA sync on steps where there was nothing to relieve. Returns `None` if CUDA isn't available; never raises, safe to call unconditionally even in a CPU-only run.

**`adaptive=True`** additionally keeps a rolling history (`trend_window` readings) of `free_fraction` and fits a simple linear trend across them. If that trend predicts crossing `min_free_fraction` within the next `lookahead_checks` calls, it clears **now**, before the threshold is actually crossed — catching a fast, steady climb before it becomes a hard OOM, instead of reacting at the same moment allocation pressure is highest. This is a linear extrapolation over your own recent readings, not a model of the allocator internals — a sudden single-step spike won't be predicted in advance, only a trend visible across several checks. `MemoryPressureEvent.predictive=True` means the trend logic fired; `predictive=False` means it was the plain reactive threshold (the only kind `adaptive=False` ever produces).

```python
cache_guard = WinCore.memory.CacheGuard(min_free_fraction=0.10)
for step, batch in enumerate(loader):
    train_step(batch)
    if step % 50 == 0:
        cache_guard.check()

# or, letting it catch a climbing trend before the hard threshold:
cache_guard = WinCore.memory.CacheGuard(min_free_fraction=0.10, adaptive=True)
```

### `to_device_non_blocking(tensor, device, pin_memory_used=True)`
`tensor.to(device, non_blocking=True)` only actually overlaps with compute if the source tensor is in **pinned** host memory — with a non-pinned tensor it silently behaves like a blocking copy (no error, just no overlap, easy to miss). This is a documented reminder wired to that actual precondition.

### `trim_working_set() -> bool`
Windows-only: calls the real Win32 `SetProcessWorkingSetSize(handle, -1, -1)` ("trim to minimum now" sentinel) to release already-freed pages from the process's working set back to the OS. Fixes the "Task Manager still shows high RAM long after a big batch/preprocessing step finished and was garbage collected" symptom — Windows doesn't proactively trim a process's working set the way Linux's allocator/kernel do, so freed-by-Python memory can sit in the reported working set until external pressure forces a trim. **Not** a fix for actual memory still in use, and not something to call every step (trimmed pages refault on next touch, which has a real cost) — call it at natural low-memory points (between epochs, after a big `del`). Returns `False` (a real no-op) on non-Windows. Raises `WorkingSetTrimError` only if the Win32 call itself reports failure.

### `estimate_worker_ram_multiplier(num_workers: int) -> str`
Explains a separate, also-real Windows RAM trap: `spawn`-based DataLoader workers (used on Windows, unlike Linux's `fork()`) each get an independent full copy of anything your `Dataset` holds in memory, instead of sharing it copy-on-write — roughly `(num_workers + 1)x` RAM for that structure. Returns a short guidance string (not a fabricated byte estimate, since this module can't know your dataset's actual size) meant to be printed alongside `recommended_dataloader_kwargs()`'s `num_workers`.

### `WorkingSetTrimError(RuntimeError)`
Raised by `trim_working_set()` only when the Win32 call itself fails.

### Dataclasses
`DataLoaderPlan(num_workers, pin_memory, persistent_workers, prefetch_factor, reason)` · `MemoryPressureEvent(free_gb, total_gb, free_fraction, cleared, predictive=False)`.

---

## `WinCore.cache`

### `DiskCache(directory, max_bytes=None, free_space_fraction=0.5)`
LRU, byte-budgeted disk cache for expensive-to-compute, cheap-to-store sample data (preprocessed tensors, decoded images, tokenized text), with Windows-safe atomic writes via `WinCore.io.atomic_write` internally — a killed process never leaves a corrupt cache entry.

**`max_bytes`**, if left as `None`, is NOT a fixed hardcoded number (previously it defaulted to a fixed 10GB) — it's computed from the actual free space on `directory`'s drive at construction time, via `shutil.disk_usage`: `free_space_fraction` (default 50%) of whatever's currently free. This is measured once, at creation — it does not shrink the cache later if the drive fills up from something else. Always pass an explicit `max_bytes` if you want a specific, predictable number instead (e.g. to leave headroom for other things writing to the same drive) — an explicit value always wins over auto-sizing.

| method | what it does |
|---|---|
| `get_or_compute(key, compute_fn) -> Any` | returns the cached value for `key` if present, else calls `compute_fn()` (no arguments — close over what you need, e.g. `lambda: self._load(idx)`), stores it, and returns it |
| `clear() -> None` | removes every cached entry and resets on-disk size tracking (hit/miss counters describe process history and are left alone) |
| `__len__()` | number of entries currently cached |
| `.stats` | a `CacheStats` (see below) |

```python
cache = WinCore.cache.DiskCache("D:/wincore_cache", max_bytes=20 * 1024**3)

# or let it size itself to the drive's free space (50% of whatever's free, by default):
cache = WinCore.cache.DiskCache("D:/wincore_cache")

class MyDataset(torch.utils.data.Dataset):
    def __getitem__(self, idx):
        return cache.get_or_compute(idx, lambda: self._load_and_preprocess(idx))
```

### `CacheStats`
`hits`, `misses`, `evictions`, `bytes_on_disk`, plus `.hit_rate() -> float`.

---

## `WinCore.kv`

Generic keyed store for per-step tensor state — **not** attention/LLM-specific. The pattern ("carry forward some tensor across sequential steps instead of recomputing it") is the same for LLM attention KV, RNN/LSTM hidden state, GNN/GAT/GIN node embeddings between propagation rounds, or PPO rollout state; this module implements the storage/eviction/compression mechanics once, generically, and leaves the model-specific math (what to store, when) to your model code — same division of responsibility as `WinCore.cache` being explicit about `.get_or_compute()` rather than monkeypatching anything.

### `StepCache(max_len=None, compress=False)`
| param | meaning |
|---|---|
| `max_len` | if set, `mode="append"` drops the oldest entries along `dim` once a key's tensor exceeds this length (sliding-window KV eviction). `None` (default) = unbounded; caller bounds it via `.clear()`. |
| `compress` | if `True`, every stored tensor is kept as `Fp8Tensor` (via `WinCore.precision.quantize_fp8`) and transparently dequantized on `.get()`. Needs the same Hopper/Ada+ GPU + torch 2.1+ as `quantize_fp8` — raises the same clear error rather than a silent uncompressed fallback. |

| method | what it does |
|---|---|
| `update(key, tensor, mode="append", dim=-2)` | `mode="append"`: `torch.cat` onto the existing tensor along `dim`, then applies `max_len` eviction if set. `mode="replace"`: overwrites entirely. `dim` defaults to `-2` (the sequence axis in the usual `[batch, heads, seq, head_dim]` KV layout) — override for other layouts, e.g. `dim=0` for a flat `[seq, hidden]` stack. |
| `get(key)` | returns the current tensor for `key` (dequantized if `compress=True`), or `None` if unset. |
| `key in cache` | membership check. |
| `keys()` | all currently-stored keys. |
| `clear(key=None)` | drops one key's state, or all state if `key` is `None` — call between independent sequences/episodes so state doesn't leak. |

```python
# Attention KV, sliding window, compressed:
kv_cache = WinCore.kv.StepCache(max_len=4096, compress=True)
kv_cache.update("layer0.k", new_k, mode="append")
kv_cache.update("layer0.v", new_v, mode="append")
k, v = kv_cache.get("layer0.k"), kv_cache.get("layer0.v")

# RNN hidden state, replaced each step:
state_cache = WinCore.kv.StepCache()
state_cache.update("gru_layer0", h_t, mode="replace")

# GNN node embeddings between propagation rounds:
gnn_cache = WinCore.kv.StepCache()
gnn_cache.update(f"round_{r}", node_embeddings, mode="replace")
```

What this is NOT: not an attention implementation (no masking/reshaping beyond a plain `torch.cat`), not automatic (no `nn.Module.forward` monkeypatching — call `.update()`/`.get()` explicitly), not disk-backed (see `WinCore.cache` for that; this is in-memory/on-device for one generation or training step's lifetime).

---

## `WinCore.kernels`

### `fused_bias_gelu(x, bias) -> torch.Tensor`
Fused `(x + bias) -> GELU`, forward and backward, as a real single-kernel-launch CUDA extension (compiled with `nvcc` via `torch.utils.cpp_extension`, not Triton — so Triton's lack of official Windows support doesn't apply here). Native for `float32`/`float64`/`float16`/`bfloat16`. For fp8 (`float8_e4m3fn`/`float8_e5m2`) it transparently upcasts to float32, runs the fused kernel, and casts back — correct result, but without the fusion speedup for that specific dtype (and always routes through fusion regardless of size, since the upcast itself already costs a full extra pass either way). Autograd works normally either way.

If the CUDA extension can't be built/loaded on this machine at all (missing CUDA Toolkit, ninja, or MSVC `cl.exe`), it falls back to an unfused but numerically identical plain-PyTorch implementation, with a one-time warning — the rest of WinCore keeps working without a native build toolchain.

**Overhead-aware size dispatch:** below a threshold element count, `fused_bias_gelu` skips the compiled kernel entirely and runs the plain-PyTorch path directly — silently, since this is a routine choice, not a degraded state. This exists because the fused kernel is a *bandwidth* optimization: for small tensors, the fixed cost of the custom kernel launch (plus the Python→C++ extension call) can outweigh what fusion saves. This isn't hypothetical — it's what the very first real benchmark against this file showed on an actual RTX 3060 (reference 0.0300 ms vs. fused 0.0664 ms — fused was *slower*, 0.45×, for that tensor size).

Check `kernel_status()` to see which path actually ran before relying on any speed claim, and `current_fusion_threshold()` / `calibrate_fusion_threshold()` to see or tune the size-based dispatch.

### `kernel_status() -> KernelStatus`
Inspects which backend is (or would be) used, **without** forcing a build — use this in benchmarks/tests to avoid asserting a speedup that can't be true on the fallback path. `KernelStatus(backend: str, reason: str | None)`.

### `current_fusion_threshold() -> int`
The element-count threshold `fused_bias_gelu` is currently using to decide whether to call the compiled kernel at all. Resolution order: a value set by `calibrate_fusion_threshold()` this process (highest priority — an actual on-machine measurement) > the `WINCORE_FUSED_MIN_ELEMENTS` environment variable > a conservative untuned default (1,048,576 elements / 1M).

### `calibrate_fusion_threshold(num_features=4096, max_rows=8192, trials=15, warmup=5) -> int`
Actually benchmarks the fused vs. unfused path on **this machine**, at increasing tensor sizes, and finds the real crossover point — then sets that as the threshold `fused_bias_gelu()` uses for the rest of this process. Requires a CUDA GPU and a machine that can build the extension; raises `RuntimeError` if either is missing, rather than silently returning an unmeasured guess. Does real GPU work and can take a few seconds — run it once per machine/GPU model you plan to support, not on every process start; cache the result (e.g. via `WINCORE_FUSED_MIN_ELEMENTS`) if startup latency matters.

The default threshold is a starting heuristic, not a hardware-measured constant — kernel launch overhead vs. memory bandwidth varies by GPU generation, driver, and CUDA Toolkit version, none of which can be honestly benchmarked without a real GPU on the machine in question.

```python
WinCore.kernels.calibrate_fusion_threshold()  # once, on each target GPU
```

### `FusedBiasGELU(num_features)`
`nn.Module` drop-in replacement for `x = x + self.bias; x = F.gelu(x)`.

### `WinCore.kernels.build.build(clean=True)`
Compiles the extension via `torch.utils.cpp_extension.load(...)`, sourcing `fused_bias_gelu_kernel.cu` from this package (bundled as package-data). Requires, on Windows: `pip install ninja`, a matching CUDA Toolkit, and MSVC's `cl.exe`.

**`cl.exe` no longer requires manually opening "x64 Native Tools Command Prompt for VS"** — `build()` now tries automatic detection first: it locates the Visual Studio install via `vswhere.exe` (Microsoft's own supported lookup mechanism), finds `vcvars64.bat` under it, runs it, and imports the resulting `PATH`/`INCLUDE`/`LIB`/`LIBPATH` into the current process — so a plain PowerShell window, or an IDE's integrated terminal (VSCode, Cursor, PyCharm, ...), works without a special launcher. Only falls through to the manual "open that special prompt" instructions if auto-detection genuinely can't find a working MSVC install at all (e.g. the C++ Build Tools component isn't installed). `clean=True` (default) wipes the previous build directory first, since a stale partial build artifact from a failed attempt being silently reused was a real bug hit during development.

```bash
pip install ninja
python -m WinCore.kernels.build
```

---

## Everything not listed here

Private helpers (leading underscore) exist in every module and are intentionally not covered — they're implementation details, not stable API, and can change without notice. If something you're calling isn't in this document, it's either private or missing from this doc by omission — open an issue / ask, rather than assuming a `_`-prefixed name is safe to depend on.
