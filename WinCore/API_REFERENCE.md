# WinCore Foundation API Reference

Every public function and class in WinCore Foundation, what it takes, what it returns, and what it actually does. `README.md` is the pitch and quick-start; this is the reference you check when you need to know a specific parameter or behavior. Nothing below is invented — it's pulled directly from the docstrings and logic in the source, and cross-checked against the actual code and test suite as of 0.8.2 (257 tests passing).

Import name is `WinCore` throughout this document (`import WinCore`) — the distribution/PyPI name is `WinCore-Foundation`; see README.md's naming note.

Anything starting with `_` (e.g. `_default_reserve`, `_ram_info`) is a private helper, not part of the public API, and isn't listed here.

---

## `WinCore` (top level)

### `WinCore.docs() -> str`
Returns the full text of `README.md`, which ships inside the installed package specifically so it survives `pip install` (a plain `readme = "README.md"` entry in `pyproject.toml` only embeds it into the PyPI page metadata — it does not, by itself, leave a browsable file in `site-packages`). Falls back to a pointer message if the README file is somehow missing from this build.

```python
import WinCore
print(WinCore.docs())
```

Also re-exported directly: `atomic_write`, `AtomicWriteError`, `atomic_torch_save`, `atomic_safetensors_save`, `fast_torch_load`, `fast_safetensors_load`, `load_checkpoint`, `safe_compile`, `should_compile`, `SafeCompiled`, `optimize`, `OptimizePlan`, `GradientAccumulator` — everything else lives under its module (`WinCore.spec`, `WinCore.cpu`, etc.), shown below.

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

### `fast_torch_load(src, device=None, mmap=True, weights_only=None, **torch_load_kwargs)`
The read-side counterpart to `atomic_torch_save`. Defaults to `torch.load(src, mmap=True, ...)` (torch>=2.1) instead of the classic full-buffer-then-unpickle path — mmap avoids the double host-RAM copy a plain `torch.load` pays (read buffer + unpickled storage) and lets the OS page cache do the work. Transparently retries WITHOUT `mmap=True` — never raises just for this — if this torch build predates the `mmap=` kwarg (`TypeError`) or the mmap attempt itself is rejected for this file/filesystem (`RuntimeError`/`OSError`, e.g. some network shares).

| param | meaning |
|---|---|
| `src` | path to the checkpoint file |
| `device` | forwarded as `map_location` (e.g. `"cuda:0"`). `None` leaves torch's own default |
| `mmap` | try the mmap path first (default `True`); falls back automatically, see above |
| `weights_only` | forwarded to `torch.load` only if given explicitly — left `None` (default) doesn't override torch's own version-dependent default for this flag |
| `**torch_load_kwargs` | forwarded to `torch.load` verbatim |

Returns exactly what `torch.load` returns.

```python
state = WinCore.io.fast_torch_load("checkpoint.pt", device="cuda:0")
```

### `fast_safetensors_load(src, device=None) -> dict`
Ergonomic wrapper around `safetensors.torch.load_file()`: same `device=` convenience (loads tensors directly onto that device instead of landing on CPU first) as `fast_torch_load`, and the same clear `ImportError` (naming the exact `pip install` line) as `atomic_safetensors_save` if the optional dependency isn't installed. Returns a flat `dict[str, torch.Tensor]` — safetensors files are always a flat tensor map by format design.

```python
tensors = WinCore.io.fast_safetensors_load("weights.safetensors", device="cuda:0")
```

### `load_checkpoint(src, device=None, mmap=True, **kwargs)`
Format-dispatching convenience wrapper: routes to `fast_safetensors_load` (if `src`'s extension is `.safetensors`, case-insensitive) or `fast_torch_load` (anything else), so resume code doesn't need its own if/else on file extension. `mmap=` is forwarded only to the torch path (ignored, not an error, on the safetensors path — there's no non-mmap mode to choose there). `**kwargs` forwarded to whichever loader is chosen; a kwarg the chosen loader doesn't accept raises the same `TypeError` calling it directly would.

```python
state = WinCore.io.load_checkpoint("checkpoint.pt", device="cuda:0")          # -> fast_torch_load
tensors = WinCore.io.load_checkpoint("weights.safetensors", device="cuda:0")  # -> fast_safetensors_load
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

### `apply(total=None, reserve=None, threads=None, set_env=True, priority=None, affinity=False, numa_aware=True, strict=False) -> ThreadPlan`
Computes the plan (same params as above) **and applies it**, best-effort:
- `torch.set_num_threads(...)` if torch is importable
- `OMP_NUM_THREADS` / `MKL_NUM_THREADS` env vars if `set_env=True` — but only takes effect on NumPy/MKL/OpenMP libraries if set *before* those libraries were imported; setting them afterward has no effect on the current process (a Python-level limitation, not a bug here)
- if `priority` is set (e.g. `"above_normal"`), calls `set_priority(priority)`
- if `affinity=True`, calls `pin_affinity(cpus)` where `cpus` comes from the same P-core-aware, NUMA-aware selection `pin_affinity()` uses by default (`_select_pin_cpus()`, sized to `plan.recommended`) — not a bare `range(plan.recommended)`
- `numa_aware` (only consulted when `affinity=True`, default `True`): on a multi-NUMA-node machine, restricts the affinity pin to node 0's CPUs BEFORE P-core selection, so a pin doesn't scatter threads across a NUMA boundary (real, measurable remote-memory latency) as a side effect

The `priority`/`affinity` calls are real OS-scheduler-level requests (see `set_priority`/`pin_affinity` below) — they reach the Python main loop and DataLoader worker processes, which thread-pool sizing alone cannot. Failures there (missing `psutil`, OS denial) are non-fatal by default and land in `plan.warnings`; pass `strict=True` to raise `PriorityError` instead.

Returns the applied `ThreadPlan` so you can log it.

```python
plan = WinCore.cpu.apply(priority="above_normal", affinity=True)
print(plan)            # ThreadPlan(total_logical=12, reserved=2, recommended=10, vendor='intel')
print(plan.warnings)   # () if everything applied cleanly
```

### `cpu_vendor() -> str`
Best-effort CPU vendor: `"amd"`, `"intel"`, or `"unknown"`. Reads `platform.processor()`'s CPUID-derived vendor string (`GenuineIntel`/`AuthenticAMD`) — not a model-number guess. Never raises; returns `"unknown"` for anything unrecognized (including most non-Windows platforms, where `platform.processor()` often returns an empty string).

### `numa_node_count() -> int`
Number of NUMA nodes Windows reports (`GetNumaHighestNodeNumber`). Returns `1` (not an error) if this can't be determined — non-Windows, an OS build without the API, or any ctypes failure. Relevant to multi-socket Intel Xeon and AMD Threadripper/EPYC-class machines specifically, not any one vendor.

### `numa_node_cpus(node=0) -> Optional[list]`
Logical CPU indices belonging to NUMA `node` (`GetNumaNodeProcessorMaskEx`, Windows only). Returns `None` — never raises — if this can't be determined.

### `set_priority(level="above_normal", pid=None) -> str`
Sets real OS scheduling priority (`SetPriorityClass` on Windows, `nice`/`setpriority` on POSIX, via `psutil`). `level` is one of `"idle"`, `"below_normal"`, `"normal"`, `"above_normal"`, `"high"`. Pass `pid` to adjust a different process (e.g. a DataLoader worker). Raises `PriorityError` if `psutil` is missing, the level name is invalid, or the OS denies the change.

### `pin_affinity(cpus=None, pid=None) -> list`
Pins the process to specific logical CPUs (`SetProcessAffinityMask` / `sched_setaffinity`, via `psutil`). `cpus` defaults to the same P-core-aware, NUMA-aware selection `_select_pin_cpus()` computes (`GetLogicalProcessorInformationEx`-based P-core detection; falls back to `range(recommended_threads().recommended)` when core-type info isn't available). Not supported on macOS (no public affinity API there) — raises `PriorityError` in that case, same as any other OS/permission denial.

### `PriorityError(RuntimeError)`
Raised by `set_priority`/`pin_affinity` (and re-raised by `apply(..., strict=True)`) on missing `psutil`, an invalid level, an unsupported platform, or an OS/permission denial.

### `ThreadPlan`
Frozen dataclass: `total_logical: int`, `reserved: int`, `recommended: int`, `warnings: tuple = ()`, `vendor: str = "unknown"` (see `cpu_vendor()` — informational only, does not change sizing).

---

## `WinCore.bootstrap`

### `optimize(cpu_kwargs=None, cuda_kwargs=None, apply_cuda=True) -> OptimizePlan`
One call applying `WinCore.cpu.apply()` then (if `apply_cuda=True`, default) `WinCore.precision.cuda_perf_defaults()`, in that order — the order matters, since CPU thread-pool env vars only take effect if set before torch initializes (see `apply()` above). A thin sequencing layer: every real effect already exists in `cpu`/`precision` with its own tests; this just calls them correctly, once.

| param | meaning |
|---|---|
| `cpu_kwargs` | forwarded as `**kwargs` to `WinCore.cpu.apply()`. `None` uses `apply()`'s own defaults |
| `cuda_kwargs` | forwarded as `**kwargs` to `WinCore.precision.cuda_perf_defaults()`. `None` uses that function's own defaults |
| `apply_cuda` | if `False`, skip the CUDA tuning step entirely — `.cuda` is `None` in the result, not an inert plan |

Never raises for a missing/unavailable subsystem: no CUDA device → `.cuda` is `cuda_perf_defaults()`'s own inert plan (unless `apply_cuda=False`); missing `psutil` → recorded in `.cpu.warnings`, same as calling `WinCore.cpu.apply()` directly.

```python
plan = WinCore.optimize()
print(plan.cpu, plan.cuda, plan.warnings)
```

### `OptimizePlan`
Dataclass: `cpu` (a `WinCore.cpu.ThreadPlan`, always present), `cuda` (a `WinCore.precision.CudaPerfPlan`, or `None` if `apply_cuda=False`), `warnings: List[str]` (flattened + prefixed, e.g. `"cpu: ..."` / `"cuda: ..."`, from both sub-plans).

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
Resolves a string to a torch dtype: `"fp16"`, `"bf16"`, `"fp32"`, `"fp64"`, `"fp8"`/`"fp8_e4m3"`/`"fp8_e5m2"`, or full torch names. fp8 needs PyTorch 2.1+ (raises a clear `ValueError`, not a confusing `AttributeError`, on older torch). **`"fp4"` intentionally raises `ValueError`** as a *compute dtype* — no IEEE/CUDA-native 4-bit compute type exists broadly. (For 4-bit *storage* compression specifically, see `quantize_fp4()` below — an explicitly experimental, separate technique from this function's compute-dtype resolution.)

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

### `cuda_perf_defaults(device_index=0, tf32=True, cudnn_benchmark=True, sdpa_backends=True, apply=True) -> CudaPerfPlan`
One call applying the standard, PyTorch-documented GPU performance knobs instead of every script re-copying the same lines: `torch.backends.cuda.matmul.allow_tf32`, `torch.backends.cudnn.allow_tf32`, `torch.set_float32_matmul_precision` (all controlled by `tf32=`), `torch.backends.cudnn.benchmark` (`cudnn_benchmark=`), and all three `scaled_dot_product_attention` backend toggles — flash/mem-efficient/math (`sdpa_backends=`). None of these are invented techniques; every one is a real, officially-documented torch switch.

| param | meaning |
|---|---|
| `device_index` | which GPU's compute capability to check for the TF32 decision |
| `tf32` | if `True` (default) AND compute capability >= 8.0 (Ampere+), enables TF32. On older hardware, left off with a warning explaining why — setting the flag there is a documented no-op, not silently claimed as "enabled" |
| `cudnn_benchmark` | if `True` (default), enables cuDNN algorithm autotuning/caching per input shape. Set `False` if your input shapes vary every step (cuDNN's shape-keyed cache re-autotunes on every new shape, which can be *slower* than leaving this off) |
| `sdpa_backends` | if `True` (default), enables all three SDPA backends so `F.scaled_dot_product_attention` can pick the fastest one available at call time. Inert (no-op) on a torch build without these toggles |
| `apply` | if `True` (default), actually sets the flags. `False` returns the plan that WOULD be applied without touching anything — for logging/dry-run use |

Safe with no CUDA device present — returns an inert all-`False` `CudaPerfPlan` with an explanatory `.reason`, doesn't raise.

```python
plan = WinCore.precision.cuda_perf_defaults()
print(plan.reason)   # e.g. "TF32 enabled (compute capability 8.6). cuDNN benchmark enabled. SDPA backends enabled: ['flash', 'mem_efficient', 'math']."
```

### `CudaPerfPlan`
Dataclass: `applied: bool`, `tf32: bool` (whether TF32 was actually enabled — reflects real hardware capability, not just what was requested), `cudnn_benchmark: bool`, `sdpa_backends_enabled: List[str]`, `warnings: List[str]`, `reason: str`.

### `quantize_fp8(tensor, fmt="e4m3", axis=None) -> Fp8Tensor`
Compresses a float tensor to real fp8 **storage** (as opposed to `amp()`'s compute-dtype selection). `fmt` is `"e4m3"` (default; more mantissa, better for weights/activations) or `"e5m2"` (more exponent range, better for gradients). Needs torch 2.1+ for the float8 dtypes to exist at all (same clear `ValueError` on mismatch as `resolve_dtype("fp8")`) — **does NOT require Hopper/Ada+ compute capability**: the cast itself is a storage-only op (round + pack bits) that torch supports on any device, CPU included, once the dtype exists in the build. Compute capability >= 8.9 only matters if you then run *fused fp8 arithmetic kernels* against the result (see `WinCore.kernels.fused_bias_gelu`'s fp8 upcast-bridge note) — this function stops at storage/bandwidth compression, so that requirement doesn't apply to it.

| param | meaning |
|---|---|
| `tensor` | a float16/bfloat16/float32 torch.Tensor. Raises `ValueError` if empty (0 elements) — there's no value range to scale against |
| `fmt` | `"e4m3"` or `"e5m2"` |
| `axis` | `None` (default): one global scale for the whole tensor, same as before this parameter existed. An axis index (e.g. `axis=0` for a weight matrix): one INDEPENDENT scale per position along that axis instead — real precision improvement when different rows/channels have different magnitude distributions (a single global scale forces the smaller-range ones toward the fp8 noise floor). Negative indices work the normal way (`axis=-1` is the last dimension). An all-zero row/channel is handled via a tiny epsilon added to its scale (avoids a 0/0 → NaN), not a special-cased branch. **Raises `ValueError` if out of range for `tensor`'s actual rank** (valid range: `-ndim` to `ndim-1`) — an earlier version silently accepted any integer and produced a working-looking but mislabeled result (see CHANGELOG) |

Dynamic scaling: fits the tensor's own value range into fp8's narrow representable range (10% headroom under `fp8_max`) instead of a fixed constant that's wrong for most tensors.

```python
packed = WinCore.precision.quantize_fp8(weight_matrix, fmt="e4m3", axis=0)   # per-row scale
restored = WinCore.precision.dequantize_fp8(packed)
```

### `dequantize_fp8(packed: Fp8Tensor)`
Inverse of `quantize_fp8` — restores an approximation at the original dtype. Works identically whether `packed.scale` is a plain float (whole-tensor) or a broadcastable tensor (per-axis) — no branching needed in calling code. Lossy by design (fp8's mantissa is too narrow to round-trip exactly); returns the best reconstruction available from the kept scale + fp8 data.

### `Fp8Tensor`
Dataclass: `data` (the fp8 tensor), `scale` (a `float`, or a broadcastable `torch.Tensor` if `axis` was given), `orig_dtype`, `axis: Optional[int] = None` (informational — which axis `scale` is per-position-of, if any).

### `quantize_fp4(tensor) -> Fp4Tensor`
**EXPERIMENTAL, software-only** 4-bit linear symmetric quantization: 15 levels (-7..+7), two 4-bit codes packed per byte, for a real ~8x storage reduction vs fp32. Read this before reaching for it over `quantize_fp8`:
- **No hardware tensor-core acceleration** — pure bit-packing on top of ordinary integer ops. There is no "fp4 matmul" this produces; using the result in arithmetic requires `dequantize_fp4()` first.
- **Only 15 levels total** — substantially more quantization error than fp8's several hundred finite values. Appropriate for aggressive, accepted-precision-loss scenarios (compressing rarely-touched optimizer state further, shrinking an already-fp8 KV cache), not weights/activations on an accuracy-sensitive path.
- **Linear, not NF4/AF4** — bitsandbytes/AWQ/GPTQ's non-uniform schemes are better-tuned for roughly-Gaussian weight distributions. This exists for experimentation (matching this module's own long-standing "not offered as a compute dtype... a quantization scheme, e.g. via bitsandbytes" stance), not to replace them for production inference quantization.
- **Implemented as a plain Python loop**, not vectorized — a deliberate correctness/auditability-over-throughput trade-off consistent with "experimental, not the recommended default". Measurably slower than `quantize_fp8` for a large tensor.

Raises `ValueError` on an empty (0-element) tensor, same as `quantize_fp8` — there's no value range to derive a scale from.

```python
packed = WinCore.precision.quantize_fp4(optimizer_state_tensor)
restored = WinCore.precision.dequantize_fp4(packed)
```

### `dequantize_fp4(packed: Fp4Tensor)`
Inverse of `quantize_fp4` — unpacks back to `orig_dtype` at `orig_shape`. Lossy in two stacked ways: the 15-level quantization error, plus the same scale-based reconstruction limits `dequantize_fp8` has.

### `Fp4Tensor`
Dataclass: `packed` (a `torch.uint8` tensor, half the byte count of an int8 encoding of the same element count), `scale: float`, `orig_shape: tuple`, `orig_dtype`, `numel: int` (original element count — packing may round up internally for an odd count; this is what `dequantize_fp4` actually restores).

### `safe_cast(tensor, dtype, on_overflow="warn")`
Casts `tensor` to `dtype`, but checks for float16's real, silent overflow-to-`inf` failure mode FIRST — a value past ~65504 becomes `inf` on a normal `.to(torch.float16)` call with no error, no warning, nothing obviously wrong until a NaN shows up several steps downstream.

| param | meaning |
|---|---|
| `tensor` | any torch.Tensor |
| `dtype` | target dtype (e.g. `torch.float16`) |
| `on_overflow` | `"warn"` (default): casts anyway (same result as plain `.to(dtype)`), but emits a `RuntimeWarning` naming the actual overflow first. `"raise"`: raises `OverflowError` instead of casting. `"clip"`: clamps to the dtype's finite range BEFORE casting, so the result has no `inf` at all |

Only checks dtypes with a known silent-overflow risk (`float16`, `float8_e4m3fn`, `float8_e5m2`) — casting to `bfloat16`/`float32`/`float64` skips the check entirely and behaves exactly like a plain `.to(dtype)`.

```python
safe = WinCore.precision.safe_cast(activations, torch.float16, on_overflow="clip")
```

---

## `WinCore.accumulate`

### `GradientAccumulator(accumulation_steps=1, model=None)`
Correct loss scaling + DDP `no_sync()` skipping for gradient accumulation (running several forward/backward passes on smaller micro-batches before one `optimizer.step()`, for an effectively larger batch size than fits in VRAM). Raises `ValueError` if `accumulation_steps < 1`, and `TypeError` if `accumulation_steps` isn't a plain `int` (a `bool` is also rejected, despite technically being an `int` subclass in Python) — a non-integer value like `2.5` (a realistic result of `total_batch_size / micro_batch_size` not dividing evenly) would otherwise produce an inconsistent window size that alternates between two different lengths instead of a fixed N, silently breaking the effective-batch-size guarantee this class exists to provide.

- **`.scale_loss(loss)`** — returns `loss / accumulation_steps`. Call `.backward()` on the RESULT, not the raw loss — skipping this trains at an effectively N-times-too-high learning rate with no error, no crash.
- **`.is_boundary() -> bool`** — `True` if the NEXT `.step()` call will complete the current window (read-only, doesn't advance the counter).
- **`.sync_context()`** — with DDP (`model` given, has `.no_sync()`, duck-typed): returns `model.no_sync()` on every micro-step except the boundary one, and a real synced pass on the boundary — so an N-step window pays for exactly ONE gradient all-reduce instead of N. Without DDP: always a harmless no-op (`contextlib.nullcontext()`).
- **`.step() -> bool`** — advances the micro-step counter; returns `True` exactly on the boundary micro-step (the signal to call `optimizer.step()`/`optimizer.zero_grad()`), `False` otherwise. Wraps to a new window automatically after returning `True`.
- **`.reset()`** — resets the counter to the start of a fresh window (e.g. after handling an exception mid-window).

```python
accum = WinCore.accumulate.GradientAccumulator(accumulation_steps=4, model=ddp_model)
for micro_batch in micro_batches:
    with accum.sync_context():
        loss = model(micro_batch)
        accum.scale_loss(loss).backward()
    if accum.step():
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
```

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

### `TrainingMonitor(loss_plateau_window=50, loss_plateau_min_relative_improvement=0.001, grad_explode_factor=10.0, grad_vanish_threshold=1e-7, grad_vanish_patience=20, signal_correlation_window=3, warmup_steps=None, expected_param_count=None, stall_factor=3.0, stall_min_samples=5, on_issue=None)`
Attach to a training loop to catch the "didn't crash, but something's silently wrong" class of problems, and to report where wall-clock time actually goes and how training progress is going. Every `record_*` call is cheap and safe to call every step — it only observes, calls `on_issue` if given one, and appends to an inspectable log; it never touches your model/optimizer/data and never raises on your behalf.

| new param (0.8.2) | meaning |
|---|---|
| `warmup_steps` | explicit override for how many initial `record_step_time()` calls count as `"warmup"` (see below). `None` (default) uses a heuristic based on `expected_param_count` instead |
| `expected_param_count` | optional model size hint used only by the `warmup_steps` heuristic when `warmup_steps` isn't given explicitly — bigger models get a longer default warmup window (cuDNN benchmark autotuning / `torch.compile` first-compile both genuinely take longer to settle on a larger model). An UNTUNED heuristic, stated plainly as such — pass `warmup_steps=` directly if you have a better number for your setup |
| `stall_factor` | a step is `"stalled"` (in `record_step_time()`) if it takes more than this many times the running steady-state average |
| `stall_min_samples` | minimum steady-state samples required before a slow step can be classified as `"stalled"` rather than just early noise |

Methods:

| method | what it does |
|---|---|
| `record_loss(step, loss_value) -> Optional[Issue]` | NaN/Inf detected immediately; `loss_plateau` (near-zero relative change) and `loss_regressing` (large *negative* relative change — actively diverging, not stuck) are distinct codes with distinct messages, once `loss_plateau_window` values have accumulated |
| `record_grad_norm(step, model) -> Optional[Issue]` | call **after** `.backward()`, **before** `.step()`/`.zero_grad()`; flags an explosion (norm jumps by `grad_explode_factor`×) or vanishing (stays below `grad_vanish_threshold` for `grad_vanish_patience` steps). Needs torch. |
| `record_signal(step, name, value, note=None) -> None` | feed in any external reading — GPU temp, VRAM pressure, an LR change, anything. Purely observational, never raises an Issue by itself. Any loss/gradient Issue emitted within `signal_correlation_window` steps gets annotated with whichever signals were recorded nearby — a **co-occurrence** note, not a causal claim. |
| `record_step_time(step, total_steps=None) -> PhaseStatus` | call once per training step to classify it `"warmup"` / `"steady_state"` / `"stalled"` and (given `total_steps`) get an ETA computed only from steady-state timing (warmup and stall outliers excluded, so neither skews it). A `"stalled"` step emits a `step_stall` warning `Issue`. See `PhaseStatus` below |
| `data_timer()` / `compute_timer()` | context managers: `with monitor.data_timer(): batch = next(it)` / `with monitor.compute_timer(): loss = train_step(batch)` |
| `gpu_timer()` | context manager measuring **actual GPU-clock busy time** (`torch.cuda.Event`) for the wrapped block, next to its CPU wall-clock time — catches the gap between "CPU thread is inside the compute block" and "GPU is actually executing", which a coarse utilization-percent reading (`nvidia-smi`) doesn't show. Calls `event.synchronize()` once per use (a real small stall), so it's meant for periodic sampling (e.g. every N steps), not necessarily every step. Degrades to a wall-clock-only no-op if CUDA (or torch) isn't available — never raises for that reason. |
| `bottleneck_report() -> dict` | accumulated data-wait vs. compute time; flags a DataLoader bottleneck if data-wait is ≥40% of total (a real, invisible-otherwise failure mode — loss looks fine, GPU is just idling between batches). If any `gpu_timer()` samples were taken, also includes `gpu_busy_seconds` / `gpu_block_wall_seconds` / `gpu_idle_fraction`, and flags a `gpu_launch_stall` Issue if idle time is ≥20% of the measured GPU-block wall time — identifying *that* the GPU stalled inside that block, not *why*; use `torch.profiler` for the latter. |
| `summary() -> List[Issue]` | every issue recorded so far, in order |

```python
monitor = WinCore.diagnostics.TrainingMonitor(
    on_issue=lambda i: print(i.severity, i.message),
    expected_param_count=count_params(model),
)
for step, batch in enumerate(loader):
    status = monitor.record_step_time(step, total_steps=total_steps)
    with monitor.data_timer():
        x, y = batch
    with monitor.compute_timer():
        loss = train_step(x, y); loss.backward()
    monitor.record_loss(step, loss.item())
    monitor.record_grad_norm(step, model)
    optimizer.step(); optimizer.zero_grad()
monitor.bottleneck_report()
```

### `PhaseStatus`
Returned by `record_step_time()`. Dataclass: `phase: str` (`"warmup"` | `"steady_state"` | `"stalled"`), `step: int`, `steps_recorded: int`, `elapsed_seconds: float` (since the FIRST `record_step_time()` call), `last_step_seconds: float`, `steady_state_avg_seconds: Optional[float]` (`None` until at least one non-warmup, non-stalled step), `steps_per_second: Optional[float]`, `eta_seconds: Optional[float]` (`None` unless `total_steps` was given AND steady state is known — a simple linear projection at the current rate, not schedule-aware), `reason: str`.

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

### `PinnedBufferPool(max_buffers=32)`
Reuses pinned (page-locked) CPU staging buffers by `(shape, dtype)` instead of allocating and pinning a fresh buffer every call. Pinning is a real CUDA driver call (`cudaHostAlloc`/`cudaHostRegister`) with measurable per-call overhead — a custom transfer pipeline (outside `DataLoader`'s own `pin_memory=True`, which already pools internally) that allocates a same-shaped pinned buffer every step pays that overhead for no reason.

- **`.get(shape, dtype=None) -> Tensor`** — returns a previously `.release()`-d buffer of the same `(shape, dtype)` if one exists, otherwise a fresh `torch.empty(shape, dtype=dtype, pin_memory=True)`. Contents are UNDEFINED either way (stale prior data, or uninitialized memory) — always overwrite before use. `dtype` defaults to `torch.float32`.
- **`.release(tensor) -> None`** — returns `tensor` to the pool for reuse. If this pushes the pool past `max_buffers` (across every shape/dtype combined), the least-recently-released buffer is evicted first.
- **`len(pool)`** — number of buffers currently pooled (released but not yet re-`.get()`-d).

NOT thread-safe by design (same as `CacheGuard`) — guard access yourself if multiple threads share one pool.

```python
pool = WinCore.memory.PinnedBufferPool()
buf = pool.get(batch.shape, batch.dtype)
buf.copy_(batch)
gpu_tensor = WinCore.memory.to_device_non_blocking(buf, "cuda:0")
pool.release(buf)   # available for the next same-shape get()
```

### Dataclasses
`DataLoaderPlan(num_workers, pin_memory, persistent_workers, prefetch_factor, reason)` · `MemoryPressureEvent(free_gb, total_gb, free_fraction, cleared, predictive=False)`.

---

## `WinCore.cache`

### `DiskCache(directory, max_bytes=None, free_space_fraction=0.5, lock_timeout=30.0)`
LRU, byte-budgeted disk cache for expensive-to-compute, cheap-to-store sample data (preprocessed tensors, decoded images, tokenized text), with Windows-safe atomic writes via `WinCore.io.atomic_write` internally — a killed process never leaves a corrupt cache entry.

**`max_bytes`**, if left as `None`, is NOT a fixed hardcoded number (previously it defaulted to a fixed 10GB) — it's computed from the actual free space on `directory`'s drive at construction time, via `shutil.disk_usage`: `free_space_fraction` (default 50%) of whatever's currently free. This is measured once, at creation — it does not shrink the cache later if the drive fills up from something else. Always pass an explicit `max_bytes` if you want a specific, predictable number instead (e.g. to leave headroom for other things writing to the same drive) — an explicit value always wins over auto-sizing.

**The `max_bytes` budget is enforced across every process sharing `directory`, not just the process that created a given `DiskCache` instance.** This matters specifically on Windows: `DataLoader(num_workers>0)` always uses `spawn` there (never `fork`, unlike Linux), so each worker gets its own independent `DiskCache` object with its own independent local bookkeeping. Without cross-process enforcement, N workers would each stay under `max_bytes` "correctly" on their own while the real directory grew toward roughly `N * max_bytes`, since no single worker's local view included what its siblings had written — and a cache entry a sibling worker had already written wouldn't be recognized as a hit by an instance that hadn't seen it written locally, wasting a recompute. Both are handled internally: hits are detected by checking the file on disk directly (`path.exists()`), not local state, and every eviction decision rescans the real directory under a lock file (`.wincore_cache.lock` inside `directory`) instead of trusting any one process's memory of what's there. `lock_timeout` (seconds) bounds how long a process waits for that lock before giving up on this eviction pass and trying again on the next store — it does not block indefinitely, and a slow/contended disk cannot hang training over cache tidiness. None of this needs any action from you — it's automatic as soon as multiple `DiskCache` instances (e.g. multiple DataLoader workers) point at the same `directory`.

| method | what it does |
|---|---|
| `get_or_compute(key, compute_fn) -> Any` | returns the cached value for `key` if present, else calls `compute_fn()` (no arguments — close over what you need, e.g. `lambda: self._load(idx)`), stores it, and returns it |
| `warm(keys, compute_fn, max_workers=None, on_error=None) -> WarmReport` | prepopulates entries for `keys` CONCURRENTLY (via `ThreadPoolExecutor`) instead of one `get_or_compute()` at a time — see below |
| `clear() -> None` | removes every cached entry (real directory scan, not just this process's local view — see above) and resets on-disk size tracking (hit/miss counters describe process history and are left alone) |
| `__len__()` | number of entries currently known to this process (after the most recent eviction scan; not necessarily a live directory count between eviction passes) |
| `.stats` | a `CacheStats` (see below) |

```python
cache = WinCore.cache.DiskCache("D:/wincore_cache", max_bytes=20 * 1024**3)

# or let it size itself to the drive's free space (50% of whatever's free, by default):
cache = WinCore.cache.DiskCache("D:/wincore_cache")

class MyDataset(torch.utils.data.Dataset):
    def __getitem__(self, idx):
        return cache.get_or_compute(idx, lambda: self._load_and_preprocess(idx))

# used from multiple DataLoader workers (the common real case on Windows) --
# nothing extra needed, each worker just constructs its own instance
# pointed at the same directory; the shared budget is enforced automatically:
loader = torch.utils.data.DataLoader(MyDataset(), num_workers=4)
```

### `DiskCache.warm(keys, compute_fn, max_workers=None, on_error=None) -> WarmReport`
Prepopulates cache entries for `keys` BEFORE the training loop needs them, computing multiple entries CONCURRENTLY via `ThreadPoolExecutor` instead of one at a time. Real speedup comes mostly from overlapping the GIL-releasing C-extension work most `compute_fn`s spend their time in (image decode/resize, NumPy, tokenizers), not just SSD write parallelism alone — threads, not processes, specifically because most real `compute_fn`s release the GIL during that work and a thread pool needs no pickling/no per-worker process startup cost.

| param | meaning |
|---|---|
| `keys` | iterable of cache keys (consumed once into a list — a generator is fine) |
| `compute_fn` | called as `compute_fn(key)` for each miss — note this takes `key` as an argument, unlike `get_or_compute`'s zero-arg closure |
| `max_workers` | forwarded to `ThreadPoolExecutor`. `None` uses that class's own default |
| `on_error` | called as `on_error(key, exception)` if `compute_fn` raises for a given key — warming continues for the rest either way |

Already-cached keys are skipped without calling `compute_fn` — safe to re-run `warm()` on a partially-warmed cache (e.g. resuming after an interrupted pass).

**Eviction runs exactly ONCE per `warm()` call, after every key has been written — not once per key.** `get_or_compute()`'s normal path runs `DiskCache`'s internal eviction check (a full directory rescan under a cross-process file lock, since the byte budget is enforced across every process sharing the cache directory — see `DiskCache`'s own docstring) after every single store, which is correct for one call at a time but would defeat the entire point of `warm()`'s concurrency if repeated per key: K keys would mean K serialized full-directory rescans under a lock, one after another, regardless of how many `max_workers` you asked for. `warm()` writes every key first (this part is genuinely concurrent), then runs one eviction pass at the end — so the returned `WarmReport` always reflects a cache already back under `max_bytes`, but the concurrency benefit isn't undermined by per-key lock contention. (Fixed in an audit pass after 0.8.2's initial release — see CHANGELOG for the earlier, slower behavior this replaced.)

```python
report = cache.warm(range(len(dataset)), compute_fn=lambda idx: preprocess(dataset.raw(idx)), max_workers=8)
print(f"{report.computed} computed, {report.already_cached} skipped, {report.failed} failed")
```

### `WarmReport`
Dataclass: `requested: int`, `already_cached: int`, `computed: int`, `failed: int`, `errors: list` (of `(key, Exception)` tuples, empty if `failed == 0`).

### `CacheStats`
`hits: int`, `misses: int`, `evictions: int`, `bytes_on_disk: int`, plus `.hit_rate` — a **property**, not a method (access as `stats.hit_rate`, no parentheses) — returning `hits / (hits + misses)`, or `0.0` if neither has happened yet.

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

## `WinCore.power`

### `prevent_sleep(keep_display_on=False) -> context manager`
Stops Windows from suspending the machine for as long as it's held — real OS call (`SetThreadExecutionState`), not a mouse-jiggle workaround. Use as `with WinCore.power.prevent_sleep(): ...` around a training run. `keep_display_on=True` also stops the display from turning off (`ES_DISPLAY_REQUIRED`); off by default since a black screen is fine (and saves power) for unattended training — the machine itself staying awake (`ES_SYSTEM_REQUIRED`) is what matters. Also usable without a `with` block via `.start()`/`.stop()`. No-op (not an error) on non-Windows. Raises `PowerError` if the real Win32 call itself fails.

### `check_tdr_risk() -> TdrReport`
Reads Windows' GPU driver watchdog timeout (`TdrDelay`, under `HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers`) — the OS-level timer (default ~2s) that kills and resets a CUDA kernel launch running longer than that, surfacing in PyTorch as `CUDA error: unspecified launch failure` with nothing pointing at this timer as the cause. Read-only diagnostics, not a fix — raising `TdrDelay` needs an admin registry edit plus a reboot, which this package won't do on your behalf. `TdrReport(platform_is_windows, tdr_delay_seconds, at_default_risk_level, message)` — `tdr_delay_seconds` is `2` (Windows' own default) whenever the registry value was never explicitly set, not `None`; `at_default_risk_level` is `True` whenever `tdr_delay_seconds <= 2`. On non-Windows, `platform_is_windows` is `False` and the rest is a fixed "not applicable".

### `PowerError`
Raised by `prevent_sleep()`'s `.start()` if the Win32 call itself reports failure.

---

## Everything not listed here

Private helpers (leading underscore) exist in every module and are intentionally not covered — they're implementation details, not stable API, and can change without notice. If something you're calling isn't in this document, it's either private or missing from this doc by omission — open an issue / ask, rather than assuming a `_`-prefixed name is safe to depend on.
