# WinCore

[![PyPI](https://img.shields.io/pypi/v/WinCore?label=PyPI)](https://pypi.org/project/WinCore/)
[![Python](https://img.shields.io/pypi/pyversions/WinCore?label=Python)](https://pypi.org/project/WinCore/)
[![License](https://img.shields.io/github/license/FWKMultiverse/WinCore?label=License)](https://github.com/FWKMultiverse/WinCore/blob/main/LICENSE)
[![Issues](https://img.shields.io/github/issues/FWKMultiverse/WinCore?label=Issues)](https://github.com/FWKMultiverse/WinCore/issues)

**Windows-focused utilities for reliable Python and PyTorch workflows.**

WinCore is a free and open-source Python library designed for practical
PyTorch and machine-learning workflows on Windows.

It provides Windows-aware utilities for system inspection, CPU and memory
management, PyTorch compilation and precision handling, GPU monitoring,
training diagnostics, multi-GPU workflows, tensor-state caching,
disk caching, safe checkpoint I/O, and optional native CUDA kernel
acceleration.

WinCore builds on PyTorch's own CUDA backend. It does not replace
CUDA, cuDNN, cuBLAS, or PyTorch's GPU runtime.

## Installation

```bash
pip install WinCore==0.7.1
```

Optional system and hardware helpers:

```bash
pip install "WinCore[full]"
```

PyPI:

https://pypi.org/project/WinCore/0.7.1/

## What WinCore Provides

* Windows-aware system and hardware detection
* CPU thread planning and OS-level CPU priority and affinity controls
* Safe file and model checkpoint writing
* PyTorch compilation with graceful eager fallback
* Automatic mixed-precision recommendations
* FP8 tensor storage compression
* Generic per-step tensor-state caching
* GPU temperature and memory monitoring
* Training diagnostics and numerical issue detection
* Multi-GPU and distributed-training utilities
* Windows-aware DataLoader and memory utilities
* Disk caching with safe writes
* Optional native CUDA kernel acceleration

WinCore is designed to be practical and conservative. When hardware or
environment information cannot be reliably detected, it prefers reporting
unknown or falling back safely rather than inventing results.

## What's New in 0.7.1

### OS-level CPU controls

`WinCore.cpu.set_priority()` and `WinCore.cpu.pin_affinity()` provide real
operating-system scheduler controls.

`WinCore.cpu.apply()` can also apply these controls together with normal
thread planning through `priority`, `affinity`, and `strict`.

Failures are non-fatal by default and are reported through `plan.warnings`.

### FP8 tensor compression

`WinCore.precision.quantize_fp8()` and
`WinCore.precision.dequantize_fp8()` provide dynamically scaled FP8 storage
compression.

Supported formats:

* `float8_e4m3fn`
* `float8_e5m2`

This can be used for tensor storage such as KV/state caches, activations,
and optimizer state.

This is separate from `WinCore.precision.amp()`: `amp()` selects a compute
dtype, while FP8 quantization changes tensor storage.

### Generic step-state cache

`WinCore.kv.StepCache` provides a generic keyed tensor-state cache.

It supports:

* append mode with sliding-window eviction
* replace mode
* independent keys
* optional FP8 compression
* transparent dequantization

It can be used for attention KV/state, RNN/LSTM hidden state, GNN/GAT
node embeddings, or other tensor state carried between steps.

```python
import WinCore

cache = WinCore.kv.StepCache(
    max_len=2048,
    compress=True,
)

cache.update(
    "layer0.k",
    new_keys,
    mode="append",
)

keys = cache.get("layer0.k")
```

### Windows working-set memory tools

`WinCore.memory.trim_working_set()` provides a real Windows working-set
trim operation.

`WinCore.memory.estimate_worker_ram_multiplier()` estimates the additional
RAM impact of Windows `spawn`-based DataLoader workers when an in-memory
Dataset is duplicated between processes.

### Native CUDA fused bias + GELU

`WinCore.kernels.fused_bias_gelu()` provides a real compiled CUDA extension
built with `torch.utils.cpp_extension` and `nvcc`.

It is not based on Triton.

The kernel fuses:

```text
x + bias
    ↓
GELU
```

into a single CUDA kernel launch.

Native fused execution supports:

* float32
* float64
* float16
* bfloat16

FP8 inputs are handled through an FP32 upcast bridge and then converted back.

### Size-aware CUDA dispatch

The fused kernel is not used blindly for every tensor size.

For small tensors, normal PyTorch execution can be faster because the launch
overhead of the custom CUDA extension can outweigh the benefit of fusion.

Inspect the current threshold with:

```python
WinCore.kernels.current_fusion_threshold()
```

The threshold can be calibrated on the current machine with:

```python
WinCore.kernels.calibrate_fusion_threshold()
```

It can also be controlled through:

```text
WINCORE_FUSED_MIN_ELEMENTS
```

### CUDA toolchain fallback

The native CUDA kernel requires:

* CUDA Toolkit
* Ninja
* MSVC `cl.exe` / Visual Studio Build Tools

If the extension cannot be built or loaded, WinCore falls back to an
unfused but numerically equivalent PyTorch implementation instead of
breaking the rest of the library.

Check the active backend with:

```python
WinCore.kernels.kernel_status()
```

A successful native build reports:

```text
KernelStatus(backend='cuda_extension', reason=None)
```

## Quick Start

```python
import WinCore

plan = WinCore.cpu.apply(
    priority="above_normal",
    affinity=True,
)

print(plan)

if plan.warnings:
    print("OS-level warnings:", plan.warnings)

check = WinCore.spec.meets_minimum(
    min_vram_gb=6,
    min_ram_gb=16,
)

if not check.ok:
    raise SystemExit(
        f"Machine doesn't meet requirements: {check.reasons}"
    )

dtype = WinCore.precision.recommended_dtype()

model = WinCore.safe_compile(model)

WinCore.atomic_write(
    lambda p: torch.save(model.state_dict(), p),
    "checkpoint.pt",
)
```

## Mixed Precision

```python
import WinCore

ctx = WinCore.precision.amp()

print(ctx.plan.reason)

for step, (x, y) in enumerate(loader):
    optimizer.zero_grad()

    with ctx.autocast():
        loss = model(x, y)

    ctx.scaler.scale(loss).backward()
    ctx.scaler.step(optimizer)
    ctx.scaler.update()
```

## Multi-GPU

```python
import WinCore
from torch.nn.parallel import DistributedDataParallel as DDP

plan = WinCore.multigpu.plan_distributed()

print(plan.reason)

topo = WinCore.multigpu.detect_topology()

if topo.measured:
    for link in topo.links:
        print(
            f"GPU{link.gpu_a} <-> GPU{link.gpu_b}: {link.label}"
        )
else:
    print(topo.note)

balance = WinCore.multigpu.check_gpu_balance()

if balance.warning:
    print(balance.warning)

plan = WinCore.multigpu.init_from_env(plan)

model = DDP(
    model,
    **WinCore.multigpu.ddp_kwargs(
        plan,
        find_unused_parameters=False,
    ),
)
```

## Windows Data Pipeline

```python
import WinCore
from torch.utils.data import DataLoader

dl_plan = WinCore.memory.recommended_dataloader_kwargs()

loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=dl_plan.num_workers,
    pin_memory=dl_plan.pin_memory,
    persistent_workers=dl_plan.persistent_workers,
    prefetch_factor=dl_plan.prefetch_factor,
)

cache = WinCore.cache.DiskCache(
    "D:/wincore_cache",
    max_bytes=20 * 1024**3,
)
```

## Training Diagnostics

WinCore can detect:

* NaN and Inf loss
* loss plateaus
* exploding gradients
* vanishing gradients
* DataLoader bottlenecks
* GPU launch and synchronization stalls
* correlated signals such as temperature and VRAM pressure
* the layer where a non-finite value first appears

Example:

```python
import WinCore

guard = WinCore.diagnostics.attach_nan_guards(
    model,
    on_issue=lambda issue: print(
        issue.severity,
        issue.data["module"],
        issue.message,
    ),
)

# Training...

guard.detach()
```

## VRAM-Aware Cache Clearing

```python
import WinCore

cache_guard = WinCore.memory.CacheGuard(
    min_free_fraction=0.10,
)

for step, batch in enumerate(loader):
    train_step(batch)

    if step % 50 == 0:
        cache_guard.check()
```

## CUDA Kernel Verification

Build the native extension with:

```bash
python -m WinCore.kernels.build
```

Then check the active backend:

```python
from WinCore.kernels.fused_bias_gelu import kernel_status

print(kernel_status())
```

The project has also been tested on a real Windows + CUDA + MSVC setup,
including the compiled fused CUDA kernel and its correctness tests. This is
one confirmed hardware and toolchain combination, not a guarantee for every
system.

## Documentation

This README provides the project overview and quick-start information.

For the complete public API, including functions, classes, parameters,
return values, and detailed behavior, see:

**[`wincore/API_REFERENCE.md`](wincore/API_REFERENCE.md)**

## Release Status

**WinCore 0.7.1 is the current public release.**

The project is publicly available for general use, but compatibility can
still vary across different Windows versions, GPU models, drivers, CUDA
Toolkit versions, MSVC toolsets, Python versions, and PyTorch builds.

If you encounter unexpected behavior, compatibility issues, or incorrect
results, please report them through the issue tracker:

https://github.com/FWKMultiverse/WinCore/issues

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

https://pypi.org/project/WinCore/0.7.1/

WinCore is developed by **FWK Multiverse** with a focus on practical,
reliable tooling for the Windows ecosystem.
