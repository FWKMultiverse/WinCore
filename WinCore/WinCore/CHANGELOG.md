# Changelog

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

See git history at https://github.com/FWKMultiverse/WinCore.
