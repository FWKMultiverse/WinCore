"""
WinCore.kernels — real, compiled CUDA extension kernels (not Triton).

Why Triton isn't used here
---------------------------
torch.compile's default Inductor backend depends on Triton, which has no
official Windows support (see WinCore.compile docstring). This package
sidesteps that entirely by writing raw CUDA C++ (.cu) compiled with nvcc
via `torch.utils.cpp_extension` -- the same mechanism NVIDIA's own
apex/Megatron-LM fused kernels use, and one of the officially supported
ways to extend PyTorch. nvcc + a CUDA Toolkit install work the same way
on Windows as on Linux, once MSVC Build Tools (cl.exe) are installed
alongside it -- Triton's Windows gap does not apply here.

What's genuinely here vs. what's not
-------------------------------------
`fused_bias_gelu` is a real, hand-written, compilable CUDA kernel that
fuses a bias-add and GELU activation into a single kernel launch, saving
one full VRAM round-trip vs. calling them as separate PyTorch ops. This
is a legitimate, narrow optimization technique (kernel fusion for
memory-bandwidth-bound elementwise ops) -- it is not a claim to beat
cuBLAS/cuDNN's compute-bound matmul/convolution kernels, which remain
the fastest available and are not touched here.

This module has NOT been compiled or benchmarked in the sandbox this
was written in -- there is no GPU there. `build.py` in this folder
compiles it via nvcc; you need a Windows (or Linux) machine with a CUDA
Toolkit and a working C++ compiler (MSVC Build Tools on Windows) to
build and actually benchmark it. Nothing in this package pretends
otherwise.
"""

try:
    from .fused_bias_gelu import (
        fused_bias_gelu,
        FusedBiasGELU,
        kernel_status,
        KernelStatus,
        current_fusion_threshold,
        calibrate_fusion_threshold,
    )
    __all__ = [
        "fused_bias_gelu",
        "FusedBiasGELU",
        "kernel_status",
        "KernelStatus",
        "current_fusion_threshold",
        "calibrate_fusion_threshold",
    ]
except ImportError:
    # Not built yet (no compiled extension present). Importing
    # WinCore.kernels should not crash the whole package -- it just
    # means this specific optional piece isn't available until you
    # run `python -m WinCore.kernels.build`.
    __all__ = []
