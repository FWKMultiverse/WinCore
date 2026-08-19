# Changelog

## 0.8.2 (audit addendum #2: 2 real bugs found on actual Windows + CUDA hardware, via a real `pytest tests/ -v` run)

Version stays 0.8.2 — same-version fix pass, not a new release. Found
by running the actual full test suite on a real Windows 11 + CUDA +
MSVC (VS2026) machine (Python 3.11.8, pytest 9.1.1) — not the sandbox
this package was originally developed in, which has no GPU and a
fake-torch test shim standing in for real tensor operations. Both bugs
below share the same root lesson: something that only ever passed
against a stand-in (the fake torch shim's no-op casts, or running one
test file in isolation instead of the whole suite together) can hide a
real defect that only shows up under real conditions — exactly why
real-hardware verification matters and sandbox-only testing, however
thorough, isn't the final word. Verified: full test suite still 257
passed in the sandbox (0 regressions from these fixes), and the fixes
themselves are now specifically written to be provably correct via a
faithful minimal reproduction (see below) rather than only "looks
right by inspection."

### Fixed
- **`WinCore.kernels.fused_bias_gelu` intermittently resolved to the
  submodule (a module object) instead of the function**, causing
  `TypeError: 'module' object is not callable` — but ONLY when running
  the full test suite together (`pytest tests/`); running
  `pytest tests/test_fused_bias_gelu.py` alone always passed, which is
  what made this so easy to miss. Root cause, found via a faithful
  minimal reproduction: Python's import system unavoidably binds an
  imported submodule onto its parent package's namespace under the
  submodule's own name, as a side effect of ANY import of that
  submodule — not only ones going through `WinCore/kernels/__init__.py`'s
  own lazy-loading `__getattr__` (added in the earlier 0.8.2 fix for
  bug #8.1). The submodule implementing this kernel was named
  `fused_bias_gelu.py` — IDENTICAL to the public function name
  `WinCore.kernels.fused_bias_gelu` — so ANY direct import of that
  submodule anywhere in the codebase (in this case,
  `test_fused_bias_gelu_dispatch.py`'s `from
  WinCore.kernels.fused_bias_gelu import (...)`, needed to reach a
  private helper not exposed through the lazy public names) silently
  overwrote the public `fused_bias_gelu` attribute with the submodule
  itself. Once pytest's collection phase had imported that test file
  (regardless of which order tests actually RUN in afterward), every
  later access to `WinCore.kernels.fused_bias_gelu` — including from
  `test_fused_bias_gelu.py`'s own correct, public-API access pattern —
  got the broken, poisoned binding. Fixed at the root: the submodule
  is renamed to `_fused_bias_gelu_kernel.py` (the public function name
  and the file it's implemented in no longer collide, so this class of
  bug can't happen again regardless of import order or which code
  imports the submodule directly) — see that file's own docstring for
  the full mechanism. `WinCore.kernels.fused_bias_gelu` (the public,
  documented attribute) is completely unchanged; only the internal
  filename moved. Also hardened `__getattr__` itself as defense in
  depth: it now binds every public name a submodule provides in the
  same call, not just the one that triggered it, so even if some
  future submodule ever shares a name with one of its own exports
  again, the first access (via ANY of that submodule's names) fully
  resolves all of them instead of leaving stragglers for direct
  imports to corrupt.
- **Two `quantize_fp8()` regression tests
  (`test_quantize_fp8_axis_none_matches_previous_global_behavior`,
  `test_quantize_fp8_per_axis_roundtrips_correctly`) asserted a
  near-exact round-trip (`abs_tol=1e-6`) that only ever passed against
  the sandbox's fake torch shim**, which doesn't perform real bit-level
  fp8 rounding (a numeric no-op stand-in for `.to(dtype)`). On real
  hardware, e4m3's ~3 mantissa bits produce a genuine, EXPECTED several-
  percent relative error per element (observed: ~0.8-3.2% on the
  reporting machine) — this is fp8 doing exactly what fp8 is supposed
  to do, not a library bug. The library code (`quantize_fp8`/
  `dequantize_fp8` themselves) was never wrong; only these two test
  assertions encoded an unrealistic "fp8 round-trips losslessly"
  expectation. Fixed by loosening both to `rel=0.15` — the SAME 15%
  bound the pre-existing, already-hardware-validated
  `test_quantize_fp8_roundtrip_preserves_magnitude` (in
  `test_precision.py`, unaffected by this bug) already uses, with its
  own comment explicitly noting "fp8 is lossy by design." A tolerance
  this size still catches a REAL regression — e.g. the earlier 0.8.2
  bug-fix-round bug where a scale factor came out ~128x wrong would
  still fail spectacularly against a 15% bound — it just no longer
  fails on ordinary, correct, expected fp8 rounding noise.

## 0.8.2 (audit addendum: 5 more bugs found and fixed in the 0.8.2 additions themselves)

Version stays 0.8.2 — this is a same-version audit/fix pass on the
work already released as 0.8.2, not a new release. A follow-up, deeper
audit pass specifically targeting the 10 new capabilities added
earlier in 0.8.2 (not the pre-existing pre-0.8.2 codebase, already
covered by the "4 confirmed bugs" entry further below) — checking real
concurrency behavior under stress, edge-case inputs (empty tensors,
out-of-range parameters, wrong argument types), and end-to-end
integration across every new module together in one pipeline. Verified:
full test suite 257 passed (up from 248 — 9 new regression tests added
alongside these fixes), 0 regressions; a real multi-threaded stress
test mixing `DiskCache.warm()` and `get_or_compute()` concurrently
against the same cache directory (5 threads, small byte budget forcing
real eviction) completed with zero errors and the in-memory index exactly
matching real on-disk state afterward.

### Fixed
- **`DiskCache.warm()` called a full directory rescan under a cross-
  process file lock (`_evict_if_needed()`) after EVERY SINGLE key**,
  not once per `warm()` call — for K keys, that's K serialized lock
  acquisitions + K full-directory `glob()`/`stat()` passes, actively
  working against the concurrency `warm()` exists to provide. Fixed by
  splitting the write path into `_store_no_evict()` (write + index
  update only) and running exactly ONE `_evict_if_needed()` pass after
  all keys in a `warm()` call have been written — `get_or_compute()`'s
  existing per-call eviction behavior is unchanged. There was also a
  **dead line of code**: the original intent (running eviction once,
  at the end) was written as `self._evict_if_needed()` placed AFTER
  the function's `return` statement — syntactically valid, but
  unreachable, so it silently never ran at all. Both are fixed
  together, with a regression test asserting `_evict_if_needed()` is
  called exactly once per `warm()` call regardless of key count
  (including when every key fails, and when every key was already
  cached).
- **`quantize_fp8()` and `quantize_fp4()` crashed with a confusing raw
  numpy error (`"zero-size array to reduction operation maximum which
  has no identity"`) on an empty (0-element) tensor**, instead of a
  clear, actionable message. Both now raise a `ValueError` explaining
  there's no meaningful value range to scale against.
- **`quantize_fp8(tensor, axis=<out of range>)` silently accepted an
  invalid axis instead of raising.** For a 2D tensor, `axis=5` (or any
  out-of-range value, positive or negative) didn't match any real
  dimension, so the "reduce over every dim except axis" logic silently
  reduced over ALL dimensions — functionally identical to `axis=None`
  — while still recording the caller's bogus axis value on the
  returned `Fp8Tensor.axis`. A working-looking result with wrong,
  misleading metadata, not an error. Now validates `axis` is in range
  for the tensor's actual rank (`-ndim` to `ndim-1`, matching how
  negative axis indexing works everywhere else in torch/NumPy) and
  raises a clear `ValueError` naming the valid range otherwise. Valid
  negative axis values (e.g. `axis=-1` for the last dimension) still
  work exactly as before — only genuinely out-of-range values are now
  rejected.
- **`GradientAccumulator(accumulation_steps=...)` silently accepted a
  non-integer value** (e.g. `2.5` — a realistic result of
  `total_batch_size / micro_batch_size` not landing on a whole number,
  using true division instead of floor division). The boundary check
  (`micro_step >= accumulation_steps`) then produces an INCONSISTENT
  window size that alternates instead of a fixed N (e.g. `2.5` cycles
  through 3-step, 2-step, 3-step, 2-step windows) — while
  `scale_loss()` keeps dividing by the same `2.5` every time, silently
  breaking the one correctness guarantee (a fixed, known effective
  batch size) this class exists to provide. Now raises `TypeError` for
  any non-`int` (a `bool` is also rejected, despite being a technical
  `int` subclass in Python) before it can produce inconsistent windows
  silently.

Six parts to this release: four confirmed-bug fixes (found by code
review of 0.7.6, not just doc review), plus ten genuinely new
capabilities covering every item in the original optimization list
(#1, #2, #3, #4, #5, #6, #7, #9). All backed by real, working, tested
code — not doc changes. Verified against the existing test suite: 248
passed (135 baseline + 113 new tests across all four 0.8.2 batches,
all passing), 0 regressions; the same 11 pre-existing failures + 2
collect errors in `test_fused_bias_gelu.py` remain (sandbox
limitation: the offline fake-torch shim has no `torch.nn` submodule,
present identically before this release).

### Added
- **`WinCore.precision.quantize_fp8(tensor, fmt, axis=None)`** — now
  supports optional per-axis scaling: one independent scale per
  position along `axis` instead of a single whole-tensor scale, real
  precision improvement when different rows/channels have genuinely
  different magnitude distributions (a single global scale forces
  small-range rows toward the fp8 noise floor if any other row has a
  much larger max). An all-zero row/channel is handled via a tiny
  epsilon added to its scale (guards the 0/0 -> NaN division a hard
  zero would cause) rather than a special-cased branch.
  `dequantize_fp8()` handles both the scalar (whole-tensor) and tensor
  (per-axis) scale forms identically — no calling-code branching
  needed. `axis=None` (default) is byte-for-byte the same behavior as
  before this parameter existed.
- **`WinCore.precision.quantize_fp4()` / `dequantize_fp4()`** (new) —
  EXPERIMENTAL, explicitly-not-recommended-by-default 4-bit linear
  symmetric quantization: 15 levels, two 4-bit codes packed per byte,
  a real ~8x storage reduction vs fp32. Loudly documented as NOT
  hardware-accelerated (pure bit-packing, not a compute dtype — no
  "fp4 matmul" exists here), NOT a replacement for bitsandbytes/AWQ/
  GPTQ's better-tuned non-uniform (NF4/AF4) schemes for production
  weight quantization, and intentionally implemented as a plain Python
  loop (not vectorized) — consistent with "experimental, for
  exploration" rather than a hot-path tool. The core packing/unpacking
  bit-math is factored into pure, torch-independent
  `_pack_4bit_codes`/`_unpack_4bit_codes` functions with direct unit
  coverage, same pattern this codebase already uses for other pure
  decision logic (`_predict_future_fraction`, `_decide_clear`).
- **`WinCore.precision.safe_cast(tensor, dtype, on_overflow="warn")`**
  — turns float16's well-known silent failure mode (a value exceeding
  ~65504 becomes `inf` on cast with no error, no warning, nothing
  obviously wrong until a NaN shows up several steps later) into an
  explicit, immediate signal: `"warn"` (default) emits a
  `RuntimeWarning` naming the actual overflow and still casts (matches
  plain `.to(dtype)` behavior otherwise), `"raise"` makes it a hard
  `OverflowError`, `"clip"` clamps to the dtype's finite range first
  so the result is guaranteed non-`inf`. Only checks dtypes with a
  known silent-overflow risk (`float16`, `float8_e4m3fn`,
  `float8_e5m2`) — casting to `bfloat16`/`float32`/`float64` skips the
  check entirely and behaves exactly like a plain `.to(dtype)`, since
  there's no overflow risk to catch there.
- **`WinCore.cpu.cpu_vendor()`** — best-effort AMD/Intel detection via
  `platform.processor()`'s CPUID-derived vendor string
  (`GenuineIntel`/`AuthenticAMD`), not a model-number guess. Surfaced
  informationally on `ThreadPlan.vendor`.
- **`WinCore.cpu.numa_node_count()` / `numa_node_cpus(node)`** — reads
  Windows' own NUMA topology (`GetNumaHighestNodeNumber` /
  `GetNumaNodeProcessorMaskEx`) for multi-socket Intel Xeon and AMD
  Threadripper/EPYC-class machines. Wired into `_select_pin_cpus()` /
  `apply(affinity=True, numa_aware=True)` (default on): on a
  multi-node machine, affinity pinning is restricted to NUMA node 0's
  CPUs BEFORE P-core selection, so a pin doesn't scatter threads
  across a NUMA boundary (real, measurable remote-memory latency) as a
  side effect of just picking "the first N indices". Falls back
  cleanly to the pre-existing behavior if node info can't be
  determined or would leave zero usable CPUs.
- **`WinCore.memory.PinnedBufferPool`** — reuses pinned (page-locked)
  CPU staging buffers by `(shape, dtype)` instead of re-pinning a
  fresh buffer every step. Pinning goes through a real CUDA driver
  call (`cudaHostAlloc`/`cudaHostRegister`) with measurable per-call
  overhead; a custom transfer pipeline (outside `DataLoader`'s own
  `pin_memory=True`, which already pools internally) that allocates a
  same-shaped pinned buffer every step pays that overhead for no
  reason, since the buffer's shape/dtype usually doesn't change even
  though its contents do. Bounded by `max_buffers` (LRU eviction) so
  it can't become an unbounded pinned-memory leak on a workload with
  many distinct shapes.
- **`WinCore.cache.DiskCache.warm(keys, compute_fn, max_workers=None)`**
  — prepopulates many cache entries concurrently (via
  `ThreadPoolExecutor`) ahead of the training loop needing them,
  instead of one `get_or_compute()` call at a time. Real speedup comes
  from overlapping the GIL-releasing C-extension work most
  `compute_fn`s are bottlenecked on (image decode/resize, NumPy,
  tokenizers), not just from SSD write parallelism alone. Skips
  already-cached keys (safe to re-run on a partially-warmed cache);
  collects per-key failures into a `WarmReport` instead of aborting
  the whole pass on one bad sample.
- **`WinCore.bootstrap.optimize(cpu_kwargs=None, cuda_kwargs=None, apply_cuda=True)`**
  — one call applying `WinCore.cpu.apply()` then
  `WinCore.precision.cuda_perf_defaults()` in the one order that
  actually matters (CPU thread-pool env vars must be set before torch
  initializes — see bug #8.1 below), returning a combined
  `OptimizePlan` instead of two separate reports to remember to check.
  A thin sequencing layer, not a new technique — every real effect
  already exists in `WinCore.cpu`/`WinCore.precision` with its own
  tests.
- **`WinCore.accumulate.GradientAccumulator`** (new module) — correct
  loss scaling for gradient accumulation (`scale_loss()`, dividing by
  `accumulation_steps` before `.backward()` — skipping this silently
  trains at an effectively N-times-too-high learning rate, with no
  error), plus DDP `model.no_sync()` skipping on every micro-step
  except the accumulation-window boundary (`sync_context()`), so an
  N-step window pays for exactly ONE gradient all-reduce instead of N
  — a real, substantial cross-GPU communication reduction for
  multi-GPU training. Duck-typed against `model.no_sync()`, so it
  works with any object exposing that method, and is a harmless no-op
  (plain synced backward) without DDP.
- **`TrainingMonitor.record_step_time(step, total_steps=None)`** (in
  `WinCore.diagnostics`) — call once per training step to classify it
  as `warmup` / `steady_state` / `stalled` and (given `total_steps`)
  get an ETA computed only from steady-state timing. `warmup_steps`
  defaults to a heuristic scaled by the new `expected_param_count`
  constructor arg (bigger models get a longer default warmup window,
  since cuDNN benchmark autotuning and `torch.compile` first-compile
  both genuinely take longer to settle on a larger model) — explicitly
  documented as an untuned heuristic, overridable via `warmup_steps=`.
  A `stalled` step (>`stall_factor`x the running steady-state average,
  only once `stall_min_samples` samples exist) emits a `step_stall`
  warning `Issue` and is excluded from the steady-state average, same
  as warmup steps are, so one bad step doesn't skew the next
  classification or the ETA.
- **`WinCore.io.fast_torch_load(src, device=None, mmap=True, ...)`** —
  fast, Windows-aware loading counterpart to `atomic_torch_save`.
  Defaults to `torch.load(..., mmap=True)` (torch>=2.1) instead of the
  classic full-buffer-then-unpickle path, which avoids the double
  host-RAM copy (read buffer + unpickled storage) that a plain
  `torch.load` pays, and lets the OS page cache do the work instead of
  Python re-copying bytes it doesn't need yet. Transparently falls
  back to the classic (non-mmap) path — never raises — if this torch
  build predates the `mmap=` kwarg (`TypeError`) or mmap itself is
  rejected for this file/filesystem, e.g. a network share
  (`RuntimeError`/`OSError`). `device=` forwards to `map_location`.
- **`WinCore.io.fast_safetensors_load(src, device=None)`** — ergonomic
  wrapper around `safetensors.torch.load_file` with the same
  `device=` convenience and the same clear, actionable `ImportError`
  (naming the exact `pip install` line) as `atomic_safetensors_save`
  if the optional dependency isn't installed.
- **`WinCore.io.load_checkpoint(src, device=None, mmap=True)`** —
  format-dispatching convenience wrapper: routes to
  `fast_safetensors_load` or `fast_torch_load` based on `src`'s file
  extension, for code paths that need to resume from either format
  without their own if/else.
- **`WinCore.precision.cuda_perf_defaults(...)`** — one call that
  applies the standard, PyTorch-documented performance knobs
  (`torch.backends.cuda.matmul.allow_tf32`,
  `torch.backends.cudnn.allow_tf32`, `torch.set_float32_matmul_precision`,
  `torch.backends.cudnn.benchmark`, and all three
  `scaled_dot_product_attention` backend toggles) instead of every
  training script re-copying the same lines from a blog post. Checks
  actual GPU compute capability first — TF32 is only enabled (and
  only reported as enabled in the returned `CudaPerfPlan`) on Ampere+
  (compute capability >= 8.0); on older hardware it's correctly left
  off with an explanatory warning rather than silently set to a flag
  that's a documented no-op there. `cudnn_benchmark` and
  `sdpa_backends` are independently toggleable (e.g.
  `cudnn_benchmark=False` for variable-input-shape workloads, where
  cuDNN's shape-keyed autotune cache works against you instead of for
  you — documented in the function's own docstring, not hidden).
  `apply=False` returns the plan that WOULD be applied without
  touching anything, for logging/dry-run use. Safe with no CUDA
  device present — returns an inert all-`False` plan, doesn't raise.
  This also serves as the real, honest answer to FP32/FP64 "training
  speed" (TF32 tensor-core matmul) — no separate FP32/FP64-specific
  addition was made in this release beyond this, since TF32 already
  covers the legitimate real-world lever there without inventing
  anything further.

### Fixed
- **`WinCore.kernels`: `import WinCore` alone silently broke
  `WinCore.cpu.apply()`'s env-var contract.** `WinCore/kernels/__init__.py`
  eagerly did `from .fused_bias_gelu import (...)`, and that module
  imports `torch` at its own top level — so the moment anything ran
  `import WinCore`, torch (and its OMP/MKL thread pools) was already
  initialized before `WinCore.cpu.apply()` ever got a chance to set
  `OMP_NUM_THREADS`/`MKL_NUM_THREADS`. `apply()`'s env vars were then
  always too late, regardless of how early in the caller's script it
  ran. Fixed by making `WinCore.kernels`'s re-exports lazy via PEP 562
  module `__getattr__` — `import WinCore` / `from . import kernels` no
  longer touches `fused_bias_gelu.py` (or torch) at all; it's deferred
  until the caller actually reaches `WinCore.kernels.fused_bias_gelu`
  or one of its sibling names. Verified: `import WinCore` no longer
  adds `torch` to `sys.modules`.
- **`WinCore.diagnostics.TrainingMonitor.record_loss()` reported
  "Loss barely moved... likely stuck" for a loss that was actually
  *diverging*.** `relative_change` is signed (positive = improving,
  negative = regressing); the plateau check only tested
  `relative_change < threshold`, which is also true for a large
  *negative* value (e.g. loss 0.687 → 3.569 gives roughly -419%
  relative change). A loss blowing up and a loss stuck flat produce
  the same warning text and code (`loss_plateau`), pointing at the
  wrong fix (LR-too-low guidance for what's actually an LR-too-high
  problem). Split into two codes: `loss_plateau` (unchanged, near-zero
  relative change) and a new `loss_regressing` (relative change
  negative and beyond the plateau threshold's own magnitude, so a tiny
  negative wobble still reads as plateau, not a false regression
  alarm) with its own message pointing at divergence causes (LR too
  high, bad batch, mixed-precision loss-scale issues).
- **`WinCore.precision.quantize_fp8()` picked the wrong fp8 ceiling
  for every spelling of "e4m3" except the exact literal string
  `"e4m3"`.** The dtype was resolved correctly via `resolve_dtype()`
  (which accepts `"e4m3"`, `"fp8_e4m3"`, `"float8_e4m3fn"` as
  synonyms), but the separate `fp8_max` ceiling used for the scale
  calculation was picked by comparing the *raw* `fmt` argument against
  the literal string `"e4m3"` only — so calling with `fmt="fp8_e4m3"`
  or `fmt="float8_e4m3fn"` correctly quantized to e4m3 storage but
  scaled against e5m2's ceiling (57344.0 instead of 448.0), off by
  ~128x, silently producing a saturated/garbage reconstruction with no
  error or warning. Fixed by deriving the ceiling from the *resolved*
  dtype instead of re-deriving it from the input string, so every
  accepted alias is correct by construction instead of needing to be
  enumerated. Verified: `quantize_fp8(t, fmt="e4m3")`,
  `fmt="fp8_e4m3"`, and `fmt="float8_e4m3fn"` now all produce the
  identical scale factor.
- **`WinCore.cache.DiskCache._evict_if_needed()` could silently drop a
  still-on-disk file from its own index.** When `unlink()` failed
  during eviction (e.g. antivirus or Windows Search Indexer holding
  the file open), the loop advanced its position counter before
  `continue`-ing — and that same counter was later used to slice which
  entries survive into the rebuilt index, so a file that failed to
  delete (and is still physically on disk) silently fell out of
  `self._index` anyway. `stats.bytes_on_disk` stayed correct (the byte
  total was never decremented for a failed unlink), but `__len__()`
  and any index-based lookup undercounted / lost track of that entry.
  Fixed by tracking evicted filenames explicitly and rebuilding the
  index from "every scanned entry not in that set," so a failed
  unlink now leaves an entry exactly where it was — present on disk
  and present in the index, instead of neither being wrong nor fully
  consistent. Verified with a forced `unlink()` failure: the file
  stays on disk and stays in `self._index` afterward.

## 0.7.6 (version bump + full documentation accuracy pass)

Requested explicitly: a thorough documentation pass checking every
module's docs against the actual current code (not just the modules
touched most recently), plus moving the package version forward to
reflect everything accumulated since 0.7.1 (posts 1-7 above, all
included in this release; the `.postN` suffixes stop here since a
release cut was requested rather than continued patch-numbering off
of 0.7.1).

### Fixed (documentation accuracy — found by diffing docs against real function signatures, not just re-reading prose)
- **`WinCore.cache.DiskCache`'s documented constructor signature was
  missing `lock_timeout`** (added in the post4 cross-process fix) —
  `API_REFERENCE.md` still showed the pre-post4 signature.
- **`CacheStats.hit_rate` was documented as a method (`.hit_rate()`)
  but is actually a `@property`** (`.hit_rate`, no parentheses) — same
  class of mistake `SafeCompiled.fell_back` already had a correctness
  note for elsewhere in the same document, just not caught here yet.
  Confirmed by grepping every `@property` in the codebase against its
  own doc entry — only this one was wrong.
- **The top-level re-export list in `API_REFERENCE.md` was missing
  `atomic_torch_save`/`atomic_safetensors_save`**, even though
  `WinCore/__init__.py` has re-exported both at the top level (`from
  WinCore import atomic_torch_save`) since before this session started
  — the doc simply never listed them.
- **`WinCore.cache`'s docs never explained the cross-process budget
  enforcement added in post4 at all** — the mechanism existed in code
  and was mentioned in passing in the README's module table, but
  `API_REFERENCE.md` (the reference for "what does this parameter
  actually do") didn't cover it. Added a full explanation, including
  what `lock_timeout` bounds and why `__len__()` reflects "as of the
  last eviction scan" rather than a always-live directory count.
- **`python -m WinCore --help`'s module list was missing `kv`**
  (pre-existing, predates this session) and, until post7, `power` —
  both now listed.
- Added a "New in 0.7.6" section to `README.md` summarizing every
  change from post4 through post7 (cross-process cache safety,
  hybrid-CPU affinity, VS2026 MSVC detection, the `apply()` integration
  fix, and `WinCore.power`) in one place, since the existing "New in
  0.7.1" section predates all of it and was never a complete picture
  of the current package on its own.
- Re-synced the duplicated doc copies (`README.md`/`API_REFERENCE.md`/
  `CHANGELOG.md` exist both at the repo root and inside `WinCore/` as
  bundled package-data) — verified byte-identical via `md5sum` after
  every edit in this pass, going forward.

No source-code behavior changes in this release — every fix here is
either a documentation correction or the version bump itself; see
0.7.1.post4 through .post7 above for the actual code changes this
release rolls up.

## 0.7.1.post7 (new module: `WinCore.power` -- sleep prevention + TDR risk detection)

Added on request: two real Windows-specific gaps for AI training that
no existing module touched, picked specifically because they're
concrete OS-level mechanisms (not vague "best practices") with no
Linux equivalent by default -- exactly the kind of thing this package
exists to paper over.

### Added
- **`WinCore.power.prevent_sleep(keep_display_on=False)`** -- a
  context manager (also usable via `.start()`/`.stop()`) around
  Windows' real `SetThreadExecutionState` API, so an unattended
  multi-hour training run doesn't get silently suspended by the OS's
  own idle timer partway through (a training loop produces none of
  the user-input "activity" that timer looks for). No-op on
  non-Windows. Raises `PowerError` only if the real Win32 call itself
  fails.
- **`WinCore.power.check_tdr_risk()`** -- reads Windows' GPU driver
  watchdog timeout (`TdrDelay` in the registry) and reports whether a
  single CUDA kernel launch has only the ~2-second OS default before
  Windows kills and resets the GPU driver. Read-only diagnostics (a
  real fix needs an admin registry edit + reboot, out of scope for an
  unprivileged training script) aimed specifically at the single most
  confusing failure mode for anyone writing/compiling custom CUDA
  kernels on Windows -- exactly the situation `WinCore.kernels` puts a
  user in -- since `CUDA error: unspecified launch failure` gives zero
  indication a Windows-specific timer, not a bug, caused it.
- 8 new tests in `tests/test_power.py`, all passing offline via mocked
  `ctypes.windll`/`winreg` (same pattern already used for
  `WinCore.memory.trim_working_set`'s tests).
- Also fixed, while touching the module list: `python -m WinCore
  --help`'s printed module list was already missing `kv` (pre-existing,
  unrelated to this change) -- added both `kv` and `power` now.

## 0.7.1.post6 (integration-layer bug: `apply(affinity=True)` bypassed its own P-core fix)

Found by tracing every call site of `pin_affinity()` after post4, not
just re-reading `cpu.py` in isolation -- a reminder that a fix at one
layer needs a check of every caller one layer up, not just the
function it lives in.

### Fixed
- **`WinCore/cpu.py`: `apply(affinity=True)` never actually got the
  P-core detection added in 0.7.1.post4.** `pin_affinity()`'s P-core
  logic only runs when its `cpus` argument is `None` -- but `apply()`
  built `list(range(plan.recommended))` itself and passed that in
  explicitly, bypassing the detection branch entirely on every single
  call. Since `apply(priority=..., affinity=True)` is the exact usage
  the README's Quick Start and `API_REFERENCE.md` show, and
  `pin_affinity()` is rarely called directly, this meant the post4 fix
  had *no effect* for essentially every real caller -- only for
  someone calling `pin_affinity()` on its own with no arguments, which
  none of the shipped tests did either (`test_pin_affinity_defaults_to
  _recommended_thread_count` calls `pin_affinity()` directly, not
  through `apply()`, so it couldn't have caught this). A unit test of
  `pin_affinity()` alone cannot catch this class of bug -- it behaves
  exactly as documented in isolation; only an assertion on what
  `apply()` itself hands to psutil would surface it. Fixed by
  extracting the P-core-vs-`range()` decision into a shared
  `_select_pin_cpus(recommended)`, used by both `pin_affinity()`'s own
  default and by `apply()` (sized to `apply()`'s own `plan.recommended`,
  which can differ from `pin_affinity()`'s default of
  `recommended_threads().recommended` if `total`/`reserve`/`threads`
  were overridden). New integration-level test
  `test_apply_affinity_uses_p_core_detection_not_bare_range` asserts
  on the actual cpu list `apply()` passes through to psutil, with
  `_detect_windows_performance_cores` mocked -- this is the class of
  test that would have caught the bug in the first place. Also fixed
  `API_REFERENCE.md`, which had documented the old bypass behavior
  (`pin_affinity(range(plan.recommended))`) as correct.

## 0.7.1.post5 (real-machine finding: MSVC auto-detection missed a preview-channel VS install)

From a real Windows 11 + Visual Studio 2026 (MSVC v145) machine:
`pytest tests/` reported 150 passed, 4 skipped -- a clean run overall,
but with a warning showing the CUDA kernel fell back to the unfused
PyTorch implementation because `cl.exe` auto-detection failed, even
though VS2026's C++ workload was genuinely installed.

### Fixed
- **`WinCore/kernels/build.py`: `_find_vcvars64()`'s `vswhere` query
  missed Visual Studio 2026 entirely because it was still on the
  Preview/Insider channel.** `vswhere -latest ...` without
  `-prerelease` only returns STABLE-channel installs -- so on a
  machine where the newest (and possibly only) VS install is a
  preview build, the query returned nothing, `_check_cl()` fell all
  the way through to the manual-fix error message, and the actionable
  advice in that message ("install Build Tools") was wrong, since
  Build Tools were already there. Fixed by adding `-prerelease` to the
  `vswhere` call -- a strict superset of the previous query, so this
  cannot cause an install that was findable before to stop being
  found. Also generalized `_check_cl()`'s error message, which
  previously named one specific toolset ("MSVC v143 - VS 2022") as if
  it were the only one that works; v143 through v145 have now all
  been confirmed fine for this kernel.

## 0.7.1.post4 (code-review findings: multi-worker cache budget, hybrid-CPU affinity, temp-file collision)

Found by a full read-through of every module (not a real-machine run
this time), specifically looking for gaps between what a module's
docstring claims to guarantee and what the implementation actually
does under conditions the existing 139-test suite didn't exercise
(multiple processes, hybrid CPUs, concurrent threads).

### Fixed
- **`WinCore/cache.py`: `DiskCache`'s `max_bytes` budget was only
  enforced against this process's own local bookkeeping, not the real
  directory.** On Windows, `DataLoader(num_workers>0)` always uses
  `spawn` (never `fork`, unlike Linux), so each worker process
  constructs its own independent `DiskCache` instance pointed at the
  same directory, with its own separate `_index`/`bytes_on_disk` that
  knows nothing about what sibling workers wrote. With N workers, each
  one independently "correctly" stayed under `max_bytes` on its own
  while the real directory could grow toward roughly N * `max_bytes`
  before eviction ever triggered -- silently, no warning, just a
  slowly filling drive. Also, a cache entry written by a sibling
  worker wasn't recognized as a hit by an instance that hadn't seen it
  written, so it was needlessly recomputed. Fixed with a new
  `_CrossProcessLock` (dependency-free, `O_CREAT|O_EXCL`-based, with
  stale-lock recovery if a process dies mid-eviction) that serializes
  eviction across every process sharing the directory, using a real
  directory rescan as the source of truth instead of local state; LRU
  order is now tracked via file mtime (bumped on every hit via
  `os.utime`, not just a local dict), which is visible to every
  process, not just the one that wrote or read the entry. Hit
  detection now checks `path.exists()` directly rather than local
  `_index` membership. 2 new tests in `tests/test_cache.py` simulating
  two independent instances sharing one directory.
- **`WinCore/cpu.py`: `pin_affinity()`'s default CPU selection could
  pin a process onto efficiency cores on Intel hybrid (P-core/E-core)
  CPUs -- the opposite of the function's own stated purpose.** The
  previous default, `range(recommended_threads().recommended)`, just
  takes the first N logical CPU *indices*. Nothing about that
  guarantees those indices are P-cores on a 12th-gen+ Intel part;
  psutil has no cross-platform notion of core type at all. Fixed by
  reading real core-type info from
  `GetLogicalProcessorInformationEx(RelationProcessorCore, ...)` via
  ctypes (Windows 10 20348+/Windows 11) and preferring detected
  P-core logical CPUs for the default pin set, falling back to the
  previous `range(N)` behavior when core-type info can't be obtained
  (non-Windows, older Windows builds, or any ctypes failure -- this
  path is defensive and never raises).
- **`WinCore/io.py`: `atomic_write()`'s temp filename could collide
  between two threads in the same process.** The temp name was
  `.{name}.tmp{pid}` -- unique per process, but two threads in the
  same process (e.g. a thread-pooled checkpoint helper) calling
  `atomic_write()` for the same destination at the same time share a
  PID and would race on the same temp file. Added the thread id to the
  temp filename.

## 0.7.1.post3 (real-machine finding: silent stale-kernel reuse on rebuild)

### Fixed
- **`WinCore/kernels/build.py`: `build(clean=True)` could silently
  return a STALE, already-loaded kernel instead of rebuilding, with no
  error at all.** Traced from a real log line on the Windows test
  machine: `"No modifications detected for re-loaded extension module
  wincore_fused_bias_gelu, skipping build step..."` appearing on a
  SECOND `build()` call within the same process, right after a first
  call had already loaded the extension. Root cause: `clean=True`'s
  cleanup was `shutil.rmtree(build_dir, ignore_errors=True)` -- and on
  Windows, a `.pyd` already loaded into the current process is
  OS-locked and can't be deleted. `ignore_errors=True` swallowed that
  `PermissionError` completely, so the directory wasn't actually
  cleaned even though `clean=True` was requested, and torch's own
  build-caching then (correctly, given what it saw) decided nothing
  needed rebuilding. Practical impact: edit
  `fused_bias_gelu_kernel.cu` and call `build()` again without
  restarting the Python process, and you'd silently get your OLD
  kernel back -- no error, no warning, nothing to indicate the edit
  wasn't picked up. Fixed with a new `_clean_build_directory()` that
  uses `shutil.rmtree(..., onerror=...)` instead of `ignore_errors=True`,
  and raises a `RuntimeWarning` naming the locked file and explaining
  the real constraint (a loaded native extension can't be hot-swapped
  in-process, full stop -- that's an OS limit, not something WinCore
  can work around) instead of pretending the clean succeeded. 4 new
  tests in `tests/test_build_clean_directory.py`, including one that
  reproduces the exact locked-file scenario via a mocked `rmtree`.

## 0.7.1.post2 (real-machine fixes, from actual Windows+CUDA+MSVC test runs)

Everything in this entry was found by actually running WinCore on a
real Windows 11 machine (RTX 3060, torch 2.6.0+cu124, CUDA Toolkit
13.1, VS2026 Build Tools) via `wincore_deep_diagnostic.py` + `pytest`,
not by reading code. All 139 pytest tests pass on that machine now
(135 passed + 4 correctly-environment-gated skips, 0 failures), and
the real compiled CUDA kernel (`fused_bias_gelu`) builds and matches
the unfused PyTorch reference across fp32/fp16/bf16/fp64/fp8.

### Fixed
- **`tests/test_cache.py`: flaky `test_no_max_bytes_auto_sizes_from_real_free_disk_space`
  and `test_free_space_fraction_is_respected`**. Reproduced on the
  real machine: failed once (`assert 69632.0 < 1024`, i.e. off by
  ~68KB out of ~64GB free), passed on an immediate retry with no code
  change. Root cause: the test called `shutil.disk_usage()` itself,
  then `DiskCache()`'s constructor called it AGAIN internally a few
  milliseconds later -- two separate real syscalls against a live,
  actively-changing disk, compared with only a 1024-byte tolerance.
  Not a `DiskCache` bug -- `_auto_size_from_free_space()`'s real
  logic (`int(free * fraction)`) is correct. Fixed by having the test
  snapshot ONE real `disk_usage()` reading and pin `shutil.disk_usage`
  to return exactly that snapshot for the test's duration (via
  `monkeypatch`), so both calls agree and the assertion can be exact
  equality instead of a tolerance -- a tolerance in KB or even MB
  doesn't fix this, it only lowers how often it's hit, since real
  background disk writes can plausibly drift by more than a few KB
  in a few milliseconds on a busy machine.

- **`WinCore/kernels/build.py`: `_check_ninja()` gave a false "not
  found" on a real machine that genuinely had `ninja==1.13.0`
  `pip install`ed**. Confirmed: `shutil.which("ninja")` failed inside
  a pytest subprocess, causing `fused_bias_gelu` to silently fall
  back to the (correct, but unfused/unaccelerated) PyTorch path with
  a misleading "Ninja is required... wasn't found on PATH" warning --
  even though the same check succeeded moments later in a different
  process on the identical machine/venv. `_check_ninja()` previously
  only ever tried `shutil.which` once. Now mirrors how `_check_cl()`
  already self-heals the equivalent MSVC problem: `_find_ninja_binary_dir()`
  locates the binary directly from the importable `ninja` Python
  package (`ninja.BIN_DIR` on newer releases; package-dir or
  `data/bin` fallbacks for older layouts) and adds that directory to
  this process's PATH before giving up. The error message, if it
  still can't find ninja anywhere, now also says WHICH of two
  different problems it is -- "not installed at all" vs. "installed
  but its binary isn't in any known layout" -- since those need
  different fixes and were previously one indistinguishable string.
  Covered by 9 new tests in `tests/test_build_ninja_detection.py`.

- **`WinCore/kernels/build.py`: `build()` now narrowly suppresses the
  upstream `_get_vc_env is private; find an alternative` `UserWarning`**
  during the actual compile (filtered by exact message text, not by
  category, so it can't accidentally swallow a real, different
  warning). This came from inside setuptools/torch's own MSVC
  detection on every real build observed so far, is already documented
  in this file's history (failure #6) as confirmed-harmless and not
  fixable from WinCore's side -- suppressing it narrowly just keeps
  `verbose=True`'s build log signal from getting buried in a warning
  that will fire on every single build, forever, regardless of whether
  anything is actually wrong.

## 0.7.1.post1 (test-infra + docs fix, not a functional release bump)

### What changed
- **`tests/conftest.py` (new)**: `tests/_fake_torch.py` was a real,
  working numpy-backed fake-torch shim -- but nothing ever imported
  it. Every test file did `torch = pytest.importorskip("torch")` at
  module scope, so on any machine/CI without real torch installed
  (this sandbox included), `test_kv.py`, `test_precision.py`,
  `test_nan_guards.py`, and both `test_fused_bias_gelu*.py` files
  skipped in full -- 0 of their assertions ever ran, silently, with
  no failure to notice. `conftest.py` now installs the fake-torch shim
  into `sys.modules["torch"]` automatically, but only when a real
  torch import fails, so real-torch machines are unaffected and
  CUDA-only tests still correctly self-skip via their own
  `torch.cuda.is_available()` guards.
- **`tests/_fake_torch.py`**: extended to unblock what `conftest.py`
  now actually exercises -- `FakeTensor.__getitem__` (slicing), a
  module-level `torch.all`, and minimal `torch.amp` / `torch.autocast`
  / `torch.cuda.amp` stand-ins so `precision.amp()`'s no-CUDA fallback
  path (construct `AmpContext`, check `.plan.enabled`/`.scaler.is_enabled()`,
  use `.autocast()` as a context manager) runs for real instead of
  being asserted only by reading the code. Still explicitly out of
  scope: `torch.nn`, autograd, and anything needing a real compiled
  CUDA kernel -- `test_fused_bias_gelu.py` and `test_nan_guards.py`
  correctly still cannot run here and say so.
- **`WinCore/precision.py` docstring fix, `quantize_fp8()`**: the
  docstring claimed this function "Requires a CUDA GPU with compute
  capability >= 8.9 (Hopper/Ada+)" -- running it for real against the
  fake-torch shim (which reports `cuda.is_available() == False`)
  showed the code never actually checks that; it only checks
  `hasattr(torch, "float8_e4m3fn")`, which is true on any torch >=2.1
  build regardless of CUDA or device. This is not a functional bug --
  the cast to a float8 dtype genuinely is a storage-only op that
  doesn't need Hopper/Ada hardware, so gating it would have been the
  wrong fix -- but the docstring overstated what was enforced. Fixed
  the docstring to describe what the code actually requires and why
  compute-capability only matters for downstream fused-fp8-compute use,
  not this function.

### How this was verified
Re-ran the full suite in the same torch-less/GPU-less/network-less
sandbox as the 0.7.1 pass above, now with the shim actually wired in:
`kv.StepCache`'s append/replace/sliding-window-eviction/independent-keys/
clear/invalid-mode logic and `precision.amp()`'s CPU-only fallback path
executed and passed for real (previously: skipped, unrun). Still not
independently verified here, same as before: anything needing a real
compiled CUDA kernel, real autograd/`torch.nn`, an actual Windows
machine, or NVLink/PCIe/NCCL hardware.

## 0.7.1

- **`cpu`**: added real OS-level `set_priority()` and `pin_affinity()`
  (Win32 `SetPriorityClass` / POSIX `nice` / `sched_setaffinity`, via
  psutil). `apply()` gained optional `priority=`, `affinity=`, and
  `strict=` args — these reach the Python main loop and DataLoader
  worker processes, which `torch.set_num_threads()`/`OMP_NUM_THREADS`
  structurally cannot (see the `WinCore.cpu` module docstring for why).
  Failures are non-fatal by default and reported in `plan.warnings`.
- **`precision`**: added `quantize_fp8()` / `dequantize_fp8()` — dynamic
  per-tensor-scaled fp8 (`e4m3`/`e5m2`) storage compression for KV
  caches, activations, or optimizer state, separate from `amp()`'s
  compute-dtype selection. Needs Hopper/Ada+ (compute capability >=
  8.9) and torch 2.1+, same requirement as `resolve_dtype("fp8")`.
- **New module `kv`**: `StepCache`, a generic keyed per-step tensor
  store (append with sliding-window eviction, or replace) usable for
  LLM attention KV, RNN/LSTM hidden state, GNN/GAT node embeddings, or
  any other model that carries tensor state across steps — not
  attention-specific. Supports transparent fp8 compression via
  `precision.quantize_fp8`.
- **`memory`**: added `trim_working_set()` (real Win32
  `SetProcessWorkingSetSize(-1, -1)` call) for the "Task Manager still
  shows high RAM long after a big batch/preprocessing step finished
  and was garbage collected" symptom — Windows doesn't proactively
  trim a process's working set the way Linux does. Added
  `estimate_worker_ram_multiplier()` explaining the separate,
  also-real issue where Windows `spawn`-based DataLoader workers each
  get a full copy of an in-memory `Dataset`, unlike Linux `fork()`'s
  copy-on-write sharing — roughly `(num_workers + 1)x` RAM.
- Version bumped 0.6.3 → 0.7.1 (skipping intermediate 0.7.0 — this
  ships all changes above together as one coherent release rather
  than splitting them).

### How this was actually verified (not just read by eye)

This sandbox has no Windows, no GPU, and no network to install real
`torch`/`pytest`. Rather than only reading the diff and asserting it's
correct, every non-CUDA-only code path above was executed for real:

- `cpu.set_priority()` / `pin_affinity()` — run for real against this
  Linux container's actual OS via `psutil` (not mocked).
- `kv.StepCache` (append growth, replace, sliding-window eviction,
  independent keys, `clear()`, invalid-mode rejection) and
  `precision.quantize_fp8`/`dequantize_fp8` (zero-tensor edge case,
  scale-vs-magnitude relationship, sign handling, e4m3/e5m2 dtype
  tagging) — executed against a small numpy-backed fake-`torch` shim
  built specifically for this (`tests/_fake_torch.py`), since no real
  torch is installable here. **This does not validate real fp8
  hardware precision loss** (numpy has no float8 dtype, so the shim's
  casts keep full precision) — it validates control flow, shapes,
  scale-factor arithmetic, and edge cases, which is where the actual
  bug this run found was:
    - **Bug found and fixed** during this pass: the *test shim*
      (`FakeTensor`) was initially missing `__bool__` and `__mul__`,
      which caused `quantize_fp8`'s zero-check branch to always
      trigger regardless of the real value — a bug in the test
      harness, not in `WinCore.precision` itself, but it silently
      passed the first run, so it's recorded here rather than glossed
      over. Fixed and re-verified; the actual `quantize_fp8` scale
      formula (`amax / (fp8_max * 0.9)`) was confirmed correct against
      its own docstring once the shim was fixed.
- `memory.trim_working_set()` — the non-Windows no-op path was run
  directly; the Windows success/failure paths were run with
  `ctypes.windll` stubbed to a fake `kernel32`, verifying the exact
  Win32 call and arguments (`SetProcessWorkingSetSize(handle, -1,
  -1)`), not just that "no exception was raised."
- Still **not** independently verified on real hardware: anything
  needing an actual CUDA device (fp8 numerical precision under real
  hardware rounding, `pin_affinity`'s effect under Windows' actual
  scheduler, `trim_working_set`'s real effect on Task Manager). Please
  run `pytest tests/` on a real Windows + GPU machine before relying
  on this in production, and open an issue if anything in this list
  turns out to be wrong.

### Known limitations / not yet done in this release

Honesty note for anyone deciding whether to upgrade: the items below
were requested but are **not** part of this release, because they need
things this environment doesn't have (a Windows machine, a real GPU,
network access to fetch dependencies) to implement and verify
responsibly rather than guess at:

- NPU-specific scheduling (DirectML/ONNX Runtime NPU execution
  providers) — no NPU hardware available to test against.
- Deeper `NestedTensorCUDA`-specific diagnostics — needs a CUDA machine
  to reproduce the actual warning/error shapes involved.
- A generalized JSON/log-format optimizer — the ask here was broad
  enough ("optimize JSON for whatever the user/developer uses it for")
  that it needs a concrete target format to design against, rather
  than a speculative implementation.
- Data-loader prefetch smoothing specifically on native Windows — the
  existing `WinCore.memory` worker-count defaults are Windows-aware,
  but deeper prefetch-pipeline changes need to be measured on a real
  Windows + spinning-disk-or-network-drive setup to know they help
  rather than just add complexity.

## 0.6.3 and earlier

See git history at https://github.com/FWKMultiverse/WinCore-Foundation.

## 0.8.2 (audit addendum #3: `build.py` never actually passed `-ccbin` to nvcc — real Windows machine, VS2026 + CUDA 13.x)

Version stays 0.8.2. `_check_cl()` always resolved and returned `cl.exe`'s real path, but `build()` discarded that return value and never passed it to nvcc. Harmless when nvcc's own auto-detection recognizes the installed MSVC toolset — but on VS2026's very new toolset with CUDA 13.x, nvcc's auto-detection didn't recognize it and failed to locate `cl.exe` even though it was genuinely on PATH, surfacing as an opaque `CreateProcess failed: The system cannot find the file specified` with no mention of `cl.exe` at all. Fixed: `build()` now passes `-ccbin <resolved cl.exe path>` to nvcc whenever `_check_cl()` resolves one, confirmed against the exact failure from a real VS2026 Developer Command Prompt + CUDA 13.3 machine. 4 new regression tests added (261 passed total, 0 regressions).
