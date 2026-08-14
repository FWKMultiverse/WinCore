# WinCore

**Windows-focused utilities for reliable Python and PyTorch workflows.**

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

**WinCore 0.6.3 is the first full public release of the project.**

The library is now publicly available for general use. However, comprehensive
testing across every supported Windows configuration, hardware combination,
GPU, driver, Python version, and PyTorch version has not yet been completed.

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
https://pypi.org/project/WinCore/0.6.3/

WinCore is developed by **FWK Multiverse** with a focus on practical,
reliable tooling for the Windows ecosystem.
