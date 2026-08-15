"""
Without this file, every test that does `torch = pytest.importorskip("torch")`
silently skips on any machine/CI without a real torch install -- including
tests/_fake_torch.py's own target modules (WinCore.kv, WinCore.precision's
non-CUDA-only paths). That shim exists specifically so this logic can be
*executed*, not just read by eye, in a torch-less sandbox -- but nothing
was importing it, so it was dead code and those modules had zero real
execution coverage here.

This installs it as `sys.modules["torch"]` ONLY when a real torch isn't
importable, so:
  - A machine with real torch: unaffected, real torch is used, real-CUDA
    tests still correctly skip via their own `if not torch.cuda.is_available()`
    guards.
  - A machine without torch (this sandbox, plain CI): test_kv.py and the
    torch-lazy parts of test_precision.py now actually run against the
    fake tensor ops instead of every test in those files silently
    disappearing into a skip count nobody reads closely.

This does NOT make CUDA-only tests (fp8 hardware rounding, real kernel
compilation, actual DDP/NCCL) runnable -- those still correctly skip or
need a real machine, and are labeled as such in the test report.
"""
try:
    import torch  # noqa: F401
except ImportError:
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    import _fake_torch

    _fake_torch.install()
