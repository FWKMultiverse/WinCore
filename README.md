# WinCore

[![PyPI](https://img.shields.io/pypi/v/WinCore?label=PyPI)](https://pypi.org/project/WinCore/)
[![Python](https://img.shields.io/pypi/pyversions/WinCore?label=Python)](https://pypi.org/project/WinCore/)
[![License](https://img.shields.io/github/license/FWKMultiverse/WinCore?label=License)](https://github.com/FWKMultiverse/WinCore/blob/main/LICENSE)
[![Issues](https://img.shields.io/github/issues/FWKMultiverse/WinCore?label=Issues)](https://github.com/FWKMultiverse/WinCore/issues)

**Windows-focused utilities for reliable Python and PyTorch workflows.**

WinCore is a free and open-source Python library designed specifically to address
common development and machine-learning workflow problems on Windows.

It provides practical utilities for system inspection, CPU and memory management,
PyTorch compilation and precision handling, GPU monitoring, diagnostics,
multi-GPU workflows, caching, and Windows-aware I/O.

## Installation

```bash
pip install WinCore==0.7.6
```

PyPI: https://pypi.org/project/WinCore/0.7.6/

## What's New in 0.7.6

**New module**
- `WinCore.power` — two Windows-specific gaps for unattended AI training
  with no default Linux equivalent:
  - `prevent_sleep(keep_display_on=False)` — a context manager (also
    usable via `.start()`/`.stop()`) around the real Win32
    `SetThreadExecutionState` API, so a multi-hour training run doesn't
    get silently suspended by Windows' idle timer, which a training loop
    never "activates" since it produces no user-input events. No-op on
    non-Windows.
  - `check_tdr_risk()` — reads the Windows GPU driver watchdog timeout
    (`TdrDelay` in the registry) and reports whether a single CUDA kernel
    launch has only the ~2-second OS default before Windows kills and
    resets the GPU driver. Read-only diagnostics aimed at the single most
    confusing failure mode when writing/compiling custom CUDA kernels on
    Windows, since `CUDA error: unspecified launch failure` otherwise
    gives no indication a Windows-specific timer, not a bug, caused it.

**Fixed — correctness**
- `WinCore.cache.DiskCache`'s `max_bytes` budget is now enforced across
  every process sharing the same cache directory, not just the process
  that created a given instance. On Windows, `DataLoader(num_workers>0)`
  always uses `spawn` (never `fork`), so each worker previously tracked
  its own independent, local view of usage — each one could "correctly"
  stay under `max_bytes` on its own while the real directory grew toward
  roughly `N × max_bytes`. Fixed with a dependency-free cross-process
  lock and real directory rescans as the source of truth instead of
  per-process local state; a new `lock_timeout` constructor parameter
  bounds how long a process waits on that lock.
- `WinCore.cpu.pin_affinity()` and `apply(affinity=True)` now both
  correctly detect real performance-core (P-core) logical CPUs on Intel
  hybrid CPUs via `GetLogicalProcessorInformationEx`, instead of assuming
  the first N logical CPU indices are the fast ones — an assumption that
  could silently pin a process onto efficiency cores on real hybrid
  layouts. `apply(affinity=True)` — the form the Quick Start actually
  uses — previously built its own CPU list and bypassed this detection
  entirely even after it was added to `pin_affinity()`; both now share
  the same selection logic.
- `WinCore.kernels.build()`'s MSVC auto-detection now finds Visual Studio
  installs on the Preview/Insider channel, not just stable releases —
  confirmed against a real Visual Studio 2026 (MSVC v145) machine where
  the previous `vswhere` query returned nothing despite the C++ Build
  Tools being genuinely installed, silently falling back to an unfused
  kernel with a misleading "Build Tools not installed" message.
- `atomic_write()`'s temporary filename could collide between two
  threads in the same process (it was unique per-process but not
  per-thread); the thread id is now included in the temp filename.
- `build(clean=True)` could silently return a stale, already-loaded
  kernel instead of rebuilding: on Windows a `.pyd` already loaded into
  the process is OS-locked and can't be deleted, and the previous cleanup
  swallowed that error instead of surfacing it. It now raises a
  `RuntimeWarning` naming the locked file and explaining the real
  constraint instead of pretending the clean succeeded.

**Fixed — documentation accuracy** (found by diffing `API_REFERENCE.md`
against the actual function signatures, not just re-reading prose)
- `DiskCache`'s documented constructor signature was missing the new
  `lock_timeout` parameter.
- `CacheStats.hit_rate` was documented as a method (`.hit_rate()`) but is
  actually a `@property` (`.hit_rate`, no parentheses).
- The top-level re-export list was missing `atomic_torch_save` /
  `atomic_safetensors_save`, even though `WinCore/__init__.py` has
  re-exported both from the top level for some time.
- `WinCore.cache`'s cross-process budget enforcement (above) wasn't
  documented at all — added a full explanation, including what
  `lock_timeout` bounds and why `__len__()` reflects "as of the last
  eviction scan" rather than a live directory count.
- `python -m WinCore --help`'s printed module list was missing `kv` and
  `power`; both are now listed.

## What WinCore Provides

* Windows-aware system and hardware detection
* CPU thread planning and resource management
* Safe file and model checkpoint writing
* PyTorch compilation with graceful fallback
* Automatic mixed-precision recommendations
* GPU temperature and memory monitoring
* Training diagnostics and numerical issue detection
* Multi-GPU and distributed-training utilities
* Windows-aware DataLoader and memory utilities
* Disk caching with safe writes
* Optional CUDA kernel acceleration
* Windows-specific power management (sleep prevention, TDR risk detection)

WinCore is designed to be practical and conservative. When hardware or
environment information cannot be reliably detected, it prefers reporting
unknown or falling back safely rather than inventing results.

## Documentation

This README provides the project overview and quick-start information.

For the complete public API, including functions, classes, parameters,
return values, and behavior, see the full API Reference:

**[`wincore/API_REFERENCE.md`](wincore/API_REFERENCE.md)**

The API Reference contains the detailed technical documentation for WinCore
and should be used when integrating or working with a specific API.

## Release Status

**WinCore 0.7.6 is the current public release.**

The library is publicly available for general use. However, comprehensive
testing across every supported Windows configuration, hardware combination,
GPU, driver, Python version, CUDA Toolkit version, MSVC toolset, and PyTorch
version has not yet been completed.

Some behavior may therefore vary between environments.

If you encounter unexpected behavior, compatibility issues, or incorrect
results, please report them through the project's issue tracker:

https://github.com/FWKMultiverse/WinCore/issues

Bug reports, reproducible examples, and environment information are especially
helpful for improving compatibility and reliability.

## Open Source

WinCore is free and open-source software released under the **MIT License**.

You are free to use, copy, modify, distribute, sublicense, and incorporate
WinCore into your own projects, subject to the terms of the MIT License.

See [`LICENSE`](LICENSE) for the complete license text.

## Support

If WinCore is useful to you and you would like to support its continued
development:

https://github.com/sponsors/FWKMultiverse

## Project

**Repository:**
https://github.com/FWKMultiverse/WinCore

**PyPI:**
https://pypi.org/project/WinCore/0.7.6/

WinCore is developed by **FWK Multiverse** with a focus on practical,
reliable tooling for the Windows ecosystem.
