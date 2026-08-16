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

- Version bump plus a full documentation-accuracy pass across the whole
  package, checking every module's docs against the actual current code
  rather than just the most recently touched modules.
- `WinCore.cache.DiskCache`'s `max_bytes` budget is now enforced across
  every process sharing the same cache directory, not just the process
  that created a given instance — fixes multi-worker `DataLoader` setups
  on Windows (which always use `spawn`, never `fork`) silently exceeding
  the intended cache size.
- `WinCore.cpu.pin_affinity()` and `apply(affinity=True)` now both detect
  real performance-core (P-core) logical CPUs on Intel hybrid CPUs,
  instead of assuming the first N logical CPU indices are the fast ones.
  `apply(affinity=True)` previously bypassed this detection; it now
  shares the same P-core-aware selection as `pin_affinity()`.
- Improved MSVC auto-detection to find Visual Studio installs on the
  Preview/Insider channel (confirmed against a real Visual Studio 2026 /
  MSVC v145 setup), not just stable releases.
- Added a new module, `WinCore.power`, with `prevent_sleep()` (stops
  Windows from suspending the machine mid-run during unattended training)
  and `check_tdr_risk()` (reads the Windows GPU driver watchdog timeout
  and flags whether a long CUDA kernel launch risks being killed and
  surfacing as a misleading `CUDA error: unspecified launch failure`).
- Corrected several `API_REFERENCE.md` inaccuracies found by diffing the
  docs against real function signatures, including a missing
  `lock_timeout` parameter on `DiskCache`, `CacheStats.hit_rate` being
  documented as a method instead of a property, and the top-level
  re-export list missing `atomic_torch_save`/`atomic_safetensors_save`.
- `python -m WinCore --help`'s printed module list now includes `kv` and
  `power`.
- No source-code behavior changes beyond what's listed above — this
  release rolls up several incremental fixes since 0.7.1 into one
  version bump plus a documentation pass.

## What's New in 0.7.1

- Added real OS-level CPU priority and CPU affinity controls through
  `set_priority()`, `pin_affinity()`, and the extended `cpu.apply()` API.
- Added FP8 tensor compression and decompression with dynamic per-tensor
  scaling.
- Added `WinCore.kv.StepCache` for generic per-step tensor state with append,
  sliding-window, replace, and optional FP8 compression support.
- Added Windows working-set trimming with `memory.trim_working_set()`.
- Added `memory.estimate_worker_ram_multiplier()` for estimating RAM usage from
  Windows `spawn`-based DataLoader workers.
- Improved native CUDA kernel build and extension handling.
- Fixed Ninja detection when Ninja is installed but its executable is not
  visible through the current process `PATH`.
- Improved Ninja diagnostics to distinguish missing installations from
  unrecognized binary layouts.
- Improved Visual Studio and MSVC detection, including preview-channel
  installations and newer MSVC toolsets.
- Fixed stale CUDA extension reuse when rebuilding a loaded kernel on Windows.
- Fixed temporary-file name collisions in `atomic_write()` between threads
  in the same process.
- Fixed `DiskCache` size accounting across multiple worker processes sharing
  the same cache directory.
- Fixed shared-cache hit detection across independent processes.
- Improved cross-process cache eviction and LRU tracking.
- Improved CPU affinity selection on Intel hybrid CPUs by preferring detected
  performance-core logical CPUs when available.
- Fixed flaky cache tests caused by multiple live disk-usage measurements
  during a single test.
- Improved test infrastructure for environments without a real PyTorch
  installation by exercising supported fallback paths instead of silently
  skipping them.
- Expanded the fake-torch test shim to cover additional tensor and AMP
  operations required by the test suite.
- Corrected the FP8 quantization documentation to match the implemented
  behavior.
- Added and expanded tests for CUDA builds, Ninja detection, clean rebuilds,
  cache behavior, CPU affinity, and atomic writes.
- Narrowly suppressed a confirmed harmless upstream MSVC environment warning
  during native CUDA builds.
- Expanded real Windows, CUDA, and MSVC verification of the compiled
  `fused_bias_gelu` kernel and related functionality.

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
