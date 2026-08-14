# WinCore

**Windows-focused utilities for reliable, practical Python and PyTorch workflows.**

WinCore is a free and open-source Python library designed specifically to address
common development and machine-learning workflow problems on Windows.

It provides practical utilities for system inspection, CPU and memory management,
PyTorch compilation and precision handling, GPU monitoring, diagnostics,
multi-GPU workflows, caching, and Windows-aware I/O.

## Installation

```bash
pip install WinCore==0.6.3
```

PyPI: https://pypi.org/project/WinCore/0.6.3/

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

WinCore is designed to be practical and conservative: when hardware or
environment information cannot be reliably detected, it prefers reporting
unknown or falling back safely rather than inventing results.

## Documentation

This README provides the project overview and quick-start information.

For the complete public API, including functions, classes, parameters,
return values, and behavioral details, see:

**[API Reference](API_REFERENCE.md)**

The API reference is the authoritative place to look when you need to use or
integrate a specific WinCore feature.

## Release Status

**WinCore 0.6.3 is the first full public release of the project.**

The library is released for general use, but comprehensive independent testing
across every supported Windows configuration, GPU, driver, Python version,
PyTorch version, and hardware combination has not yet been completed.

If you encounter unexpected behavior, compatibility issues, or incorrect results,
please report them so they can be investigated and improved.

**Bug reports:**
https://github.com/FWKMultiverse/WinCore/issues

## Open Source

WinCore is released under the **MIT License**.

You are free to use, modify, distribute, and integrate WinCore into your own
projects, subject to the terms of the license.

See [`LICENSE`](LICENSE) for the full license text.

## Support the Project

If WinCore is useful to you and you would like to support its continued
development:

https://github.com/sponsors/FWKMultiverse

## Project

**Repository:**
https://github.com/FWKMultiverse/WinCore

WinCore is developed by **FWK Multiverse** and is built with a focus on
practical, reliable tooling for the Windows ecosystem.
