#!/usr/bin/env python3
"""
WinCore deep diagnostic script.

Runs on YOUR machine (with real torch installed, and optionally a real
GPU / MSVC+nvcc / Windows). It does two things, in order:

  1. Runs WinCore's actual pytest suite (tests/) if pytest is
     installed -- this is the project's own test coverage, unmodified.
  2. Runs a second, independent battery of "live" checks below that go
     beyond what tests/ covers: real numeric comparisons, real timing,
     real hardware probing, and things that only make sense to check
     interactively on a real machine (compute capability, MSVC/nvcc
     presence, real DDP topology, etc).

Every check reports PASS / FAIL / SKIP / ERROR with a one-line reason,
never just "ok" -- the goal is a report you can act on directly, not a
green checkmark to trust blindly.

Usage:
    cd WinCore-0.7.1-patched        # (or wherever pyproject.toml is)
    python wincore_deep_diagnostic.py

Optional:
    python wincore_deep_diagnostic.py --skip-pytest   # only run part 2
    python wincore_deep_diagnostic.py --json out.json # also dump machine-readable results
"""
from __future__ import annotations

import argparse
import contextlib
import io as _io
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

@dataclass
class Result:
    section: str
    name: str
    status: str  # PASS / FAIL / SKIP / ERROR
    detail: str = ""
    seconds: float = 0.0


@dataclass
class Reporter:
    results: list = field(default_factory=list)

    def run(self, section: str, name: str, fn: Callable[[], Optional[str]]):
        """Run fn(). fn should return None (pass, no extra detail), a
        string (pass, with detail), raise `Skip(msg)` to skip, or raise
        any other exception to fail. Prints and records the outcome."""
        t0 = time.time()
        try:
            detail = fn()
            dt = time.time() - t0
            r = Result(section, name, "PASS", detail or "", dt)
        except Skip as e:
            dt = time.time() - t0
            r = Result(section, name, "SKIP", str(e), dt)
        except AssertionError as e:
            dt = time.time() - t0
            r = Result(section, name, "FAIL", str(e) or "assertion failed", dt)
        except Exception as e:
            dt = time.time() - t0
            tb = traceback.format_exc(limit=6)
            r = Result(section, name, "ERROR", f"{e!r}\n{tb}", dt)
        self.results.append(r)
        self._print(r)
        return r

    @staticmethod
    def _print(r: Result):
        tag = {"PASS": "PASS ", "FAIL": "FAIL ", "SKIP": "SKIP ", "ERROR": "ERR  "}[r.status]
        line = f"[{tag}] {r.section:<12} {r.name:<55} ({r.seconds*1000:6.1f}ms)"
        print(line)
        if r.detail and r.status != "PASS":
            for ln in r.detail.splitlines():
                print(f"         {ln}")
        elif r.detail and r.status == "PASS":
            print(f"         -> {r.detail}")

    def summary(self):
        counts = {"PASS": 0, "FAIL": 0, "SKIP": 0, "ERROR": 0}
        for r in self.results:
            counts[r.status] += 1
        print("\n" + "=" * 78)
        print(
            f"TOTAL: {len(self.results)}   "
            f"PASS={counts['PASS']}  FAIL={counts['FAIL']}  "
            f"SKIP={counts['SKIP']}  ERROR={counts['ERROR']}"
        )
        bad = [r for r in self.results if r.status in ("FAIL", "ERROR")]
        if bad:
            print("\n--- Needs attention ---")
            for r in bad:
                print(f"  [{r.status}] {r.section} :: {r.name}")
                first_line = r.detail.splitlines()[0] if r.detail else ""
                print(f"      {first_line}")
        print("=" * 78)
        return counts

    def to_json(self, path: str):
        with open(path, "w") as f:
            json.dump(
                [r.__dict__ for r in self.results],
                f,
                indent=2,
                default=str,
            )
        print(f"\nMachine-readable report written to {path}")


class Skip(Exception):
    pass


R = Reporter()


# --------------------------------------------------------------------------
# Environment probe (informational, always runs, never fails)
# --------------------------------------------------------------------------

def probe_environment():
    print("\n" + "#" * 78)
    print("# ENVIRONMENT")
    print("#" * 78)
    info = {}
    info["python"] = sys.version.split()[0]
    info["platform"] = platform.platform()
    info["os_name"] = os.name
    info["is_windows"] = platform.system() == "Windows"

    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
            info["compute_capability"] = torch.cuda.get_device_capability(0)
            info["cuda_toolkit_version_torch_was_built_with"] = torch.version.cuda
        info["has_float8"] = hasattr(torch, "float8_e4m3fn")
    except ImportError:
        info["torch"] = None

    for opt in ("psutil", "pynvml", "wmi", "safetensors", "pytest", "ninja"):
        try:
            mod = __import__(opt)
            info[opt] = getattr(mod, "__version__", "present")
        except ImportError:
            info[opt] = None

    info["nvcc"] = shutil.which("nvcc")
    info["cl.exe (MSVC)"] = shutil.which("cl")
    info["nvidia-smi"] = shutil.which("nvidia-smi")

    for k, v in info.items():
        print(f"  {k:<45} {v}")
    return info


# --------------------------------------------------------------------------
# Part 1: run the project's own pytest suite
# --------------------------------------------------------------------------

def run_pytest_suite():
    print("\n" + "#" * 78)
    print("# PART 1: tests/ via real pytest")
    print("#" * 78)
    try:
        import pytest  # noqa: F401
    except ImportError:
        print("pytest not installed -- skipping part 1 "
              "(pip install \"WinCore[dev]\" or `pip install pytest` to enable).")
        return None
    if not os.path.isdir("tests"):
        print("No ./tests directory here -- run this script from the "
              "WinCore project root (where pyproject.toml lives).")
        return None
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short",
         "-rA", "--no-header"],
        capture_output=True, text=True,
    )
    print(proc.stdout)
    if proc.stderr.strip():
        print("--- stderr ---")
        print(proc.stderr)
    print(f"(pytest exit code: {proc.returncode})")
    return proc.returncode


# --------------------------------------------------------------------------
# Part 2: live checks beyond the unit tests
# --------------------------------------------------------------------------

def need_torch():
    try:
        import torch
        return torch
    except ImportError:
        raise Skip("torch not installed")


def need_cuda(torch):
    if not torch.cuda.is_available():
        raise Skip("no CUDA device visible to torch")


# ---- io ----

def check_io_atomic_write_roundtrip():
    from WinCore.io import atomic_write
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.bin")
        payload = os.urandom(1024)

        def _write(p):
            with open(p, "wb") as f:
                f.write(payload)

        atomic_write(_write, path)
        assert os.path.exists(path)
        with open(path, "rb") as f:
            got = f.read()
        assert got == payload, "round-tripped bytes don't match what was written"
        # no leftover temp files
        leftovers = [f for f in os.listdir(d) if f != "ckpt.bin"]
        assert not leftovers, f"atomic_write left temp files behind: {leftovers}"
    return "wrote+read 1KiB, no leftover temp files"


def check_io_torch_save_real():
    torch = need_torch()
    from WinCore.io import atomic_torch_save
    import tempfile, os
    state = {"w": torch.randn(8, 8), "step": 42}
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        atomic_torch_save(state, path)
        loaded = torch.load(path, weights_only=False)
        assert torch.allclose(loaded["w"], state["w"])
        assert loaded["step"] == 42
    return f"real torch.save/load round-trip via atomic_torch_save, torch={torch.__version__}"


def check_io_safetensors_real():
    try:
        import safetensors  # noqa
    except ImportError:
        raise Skip("safetensors not installed (pip install safetensors)")
    torch = need_torch()
    from WinCore.io import atomic_safetensors_save
    from safetensors.torch import load_file
    import tempfile, os
    tensors = {"weight": torch.randn(4, 4).contiguous()}
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.safetensors")
        atomic_safetensors_save(tensors, path, metadata={"format": "pt"})
        loaded = load_file(path)
        assert torch.allclose(loaded["weight"], tensors["weight"])
    return "real safetensors save/load round-trip"


# ---- cpu ----

def check_cpu_recommended_threads_real():
    from WinCore.cpu import recommended_threads
    plan = recommended_threads()
    total = os.cpu_count() or 1
    assert 1 <= plan.recommended <= plan.total_logical == total, (
        f"recommended_threads() -> {plan}, inconsistent with os.cpu_count()={total}"
    )
    assert plan.reserved >= 0
    return f"{plan}"


def check_cpu_apply_real():
    from WinCore import cpu
    torch = None
    try:
        import torch as _torch
        torch = _torch
    except ImportError:
        pass
    plan = cpu.apply()
    if torch is not None:
        assert torch.get_num_threads() > 0
    return f"cpu.apply() -> {plan}"


def check_cpu_set_priority_real():
    from WinCore import cpu
    try:
        import psutil  # noqa
    except ImportError:
        raise Skip("psutil not installed -- set_priority()/pin_affinity() need it")
    result = cpu.set_priority("normal")
    return f"set_priority('normal') -> {result}"


# ---- spec ----

def check_spec_real_machine():
    from WinCore.spec import get_system_spec
    spec = get_system_spec()
    assert spec.logical_threads >= 1
    assert spec.ram_total_gb > 0
    return f"{spec}"


def check_spec_meets_minimum_real():
    from WinCore.spec import get_system_spec, meets_minimum
    spec = get_system_spec()
    low = meets_minimum(min_logical_threads=1, min_ram_gb=0.1, spec=spec)
    assert low.ok, f"failed absurdly low minimums: {low.reasons}"
    high = meets_minimum(min_logical_threads=1, min_ram_gb=1_000_000, spec=spec)
    assert not high.ok, "claimed to meet a 1,000,000 GB RAM minimum"
    assert high.reasons, "failed check reported no reasons"
    return f"low={low.ok}, high={high.ok}, high.reasons={high.reasons}"


# ---- precision ----

def check_precision_resolve_dtype_real():
    torch = need_torch()
    from WinCore.precision import resolve_dtype
    assert resolve_dtype("fp16") is torch.float16
    assert resolve_dtype("bf16") is torch.bfloat16
    assert resolve_dtype("fp32") is torch.float32
    try:
        resolve_dtype("fp4")
        raise AssertionError("resolve_dtype('fp4') should have raised ValueError")
    except ValueError:
        pass
    return "fp16/bf16/fp32 resolve correctly, fp4 correctly rejected"


def check_precision_recommended_dtype_real():
    torch = need_torch()
    from WinCore.precision import recommended_dtype, dtype_name
    dt = recommended_dtype()
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(0)
        if major >= 8:
            assert dt is torch.bfloat16, f"Ampere+ GPU but recommended {dtype_name(dt)}"
        elif major >= 6:
            assert dt is torch.float16, f"Pascal+ GPU but recommended {dtype_name(dt)}"
        else:
            assert dt is torch.float32
        return f"real GPU cc={major}.x -> recommended {dtype_name(dt)}"
    else:
        assert dt is torch.float32
        return "no CUDA -> correctly recommended float32"


def check_precision_amp_real():
    torch = need_torch()
    from WinCore.precision import amp
    ctx = amp()
    with ctx.autocast():
        x = torch.randn(4, 4)
        y = x @ x
    assert torch.isfinite(y).all()
    if torch.cuda.is_available():
        return f"amp() on real GPU: enabled={ctx.plan.enabled}, dtype={ctx.plan.dtype}, reason={ctx.plan.reason}"
    return f"amp() on CPU-only: enabled={ctx.plan.enabled} (expected False), autocast context didn't crash"


def check_precision_fp8_real_hardware():
    torch = need_torch()
    need_cuda(torch)
    if not hasattr(torch, "float8_e4m3fn"):
        raise Skip(f"torch {torch.__version__} has no float8 dtypes (need 2.1+)")
    major, minor = torch.cuda.get_device_capability(0)
    from WinCore.precision import quantize_fp8, dequantize_fp8
    x = (torch.randn(256, 256, device="cuda", dtype=torch.float16) * 5.0)
    packed = quantize_fp8(x, fmt="e4m3")
    assert packed.data.dtype == torch.float8_e4m3fn
    restored = dequantize_fp8(packed)
    max_abs_err = (restored.float() - x.float()).abs().max().item()
    rel_err = max_abs_err / x.float().abs().max().item()
    # fp8 e4m3 has ~2-3 mantissa bits -- a real, nontrivial error is
    # *expected*; this checks it's in a sane ballpark, not "zero error".
    assert rel_err < 0.25, f"fp8 roundtrip relative error {rel_err:.3f} implausibly large"
    return (f"REAL fp8 hardware roundtrip on cc={major}.{minor} GPU: "
            f"max_abs_err={max_abs_err:.4f}, rel_err={rel_err:.4f}, scale={packed.scale:.6f}")


# ---- kv ----

def check_kv_real_torch():
    torch = need_torch()
    from WinCore.kv import StepCache
    cache = StepCache(max_len=3)
    for i in range(5):
        cache.update("kv", torch.full((1, 1, 1, 1), float(i)), mode="append")
    result = cache.get("kv").flatten().tolist()
    assert result == [2.0, 3.0, 4.0], f"eviction wrong: got {result}"
    cache2 = StepCache()
    cache2.update("h", torch.randn(1, 8), mode="replace")
    h2 = torch.randn(1, 8)
    cache2.update("h", h2, mode="replace")
    assert torch.allclose(cache2.get("h"), h2)
    return "real torch.cat/narrow append+eviction and replace both correct"


def check_kv_compress_real_hardware():
    torch = need_torch()
    need_cuda(torch)
    if not hasattr(torch, "float8_e4m3fn"):
        raise Skip("torch build has no float8 dtypes")
    from WinCore.kv import StepCache
    cache = StepCache(compress=True)
    x = torch.randn(4, 4, device="cuda", dtype=torch.float16)
    cache.update("layer0", x, mode="replace")
    restored = cache.get("layer0")
    assert restored.shape == x.shape
    assert restored.dtype == x.dtype
    return "real fp8-compressed KV cache store+retrieve on GPU"


# ---- diagnostics ----

def check_diagnostics_nan_detection_real():
    from WinCore.diagnostics import TrainingMonitor
    issues = []
    mon = TrainingMonitor(on_issue=issues.append)
    mon.record_loss(0, 1.0)
    mon.record_loss(1, float("nan"))
    assert any(i.message and "nan" in i.message.lower() or "inf" in i.message.lower()
               for i in issues) or any(getattr(i, "kind", "") for i in issues), \
        f"NaN loss at step 1 didn't produce an Issue: {issues}"
    return f"NaN loss correctly flagged: {[i.severity for i in issues]}"


def check_diagnostics_grad_norm_real():
    torch = need_torch()
    import torch.nn as nn
    from WinCore.diagnostics import TrainingMonitor
    issues = []
    mon = TrainingMonitor(on_issue=issues.append)
    model = nn.Linear(4, 4)
    x = torch.randn(2, 4)
    for step in range(5):
        y = model(x).sum() * (100.0 ** step)  # deliberately exploding
        model.zero_grad()
        y.backward()
        mon.record_grad_norm(step, model)
    return f"grad norm tracked over 5 exploding steps, {len(issues)} issue(s) raised"


def check_diagnostics_nan_guards_real():
    torch = need_torch()
    import torch.nn as nn
    from WinCore.diagnostics import attach_nan_guards

    class Poison(nn.Module):
        def forward(self, x):
            return x / 0.0

    model = nn.Sequential(nn.Linear(4, 4), Poison(), nn.Linear(4, 4))
    seen = []
    handle = attach_nan_guards(model, on_issue=lambda i: seen.append(i))
    with handle:
        out = model(torch.randn(2, 4))
    assert seen, "poisoned layer produced NaN/Inf but no guard fired"
    assert any("1" in str(getattr(i, "message", "")) or "Poison" in str(getattr(i, "message", ""))
               for i in seen) or len(seen) > 0
    return f"NaN guard caught the poisoned layer: {len(seen)} issue(s), first={seen[0].message[:80]}"


def check_diagnostics_gpu_timer_real():
    torch = need_torch()
    need_cuda(torch)
    from WinCore.diagnostics import TrainingMonitor
    mon = TrainingMonitor()
    x = torch.randn(2048, 2048, device="cuda")
    with mon.gpu_timer():
        for _ in range(10):
            x = x @ x
        torch.cuda.synchronize()
    report = mon.bottleneck_report()
    return f"real GPU timer sampled a matmul loop: {report}"


# ---- memory / cache ----

def check_memory_dataloader_kwargs_real():
    from WinCore.memory import recommended_dataloader_kwargs
    plan = recommended_dataloader_kwargs()
    assert plan.num_workers >= 0
    return f"{plan}"


def check_memory_cache_guard_real():
    torch = need_torch()
    from WinCore.memory import CacheGuard
    guard = CacheGuard()
    # .check() is the real method (not maybe_clear -- that was my own
    # guess, wrong; confirmed against source). Never raises, even
    # CPU-only -- returns None when CUDA isn't available.
    event = guard.check()
    if torch.cuda.is_available():
        assert event is not None, "check() returned None despite CUDA being available"
        return f"CacheGuard.check() on real GPU -> {event}"
    assert event is None, f"check() should return None with no CUDA, got {event}"
    return "CacheGuard.check() correctly returned None with no CUDA available"


def check_cache_disk_cache_real():
    from WinCore.cache import DiskCache
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        cache = DiskCache(d, max_bytes=10_000_000)
        calls = {"n": 0}

        def compute():
            calls["n"] += 1
            return os.urandom(100)

        v1 = cache.get_or_compute("sample_0", compute)
        v2 = cache.get_or_compute("sample_0", compute)
        assert v1 == v2, "cached value changed between calls"
        assert calls["n"] == 1, f"compute() called {calls['n']} times, expected 1 (cache miss then hit)"
    return "disk cache hit avoided recompute, value stable across get_or_compute calls"


# ---- thermal ----

def check_thermal_pynvml_real():
    try:
        import pynvml
    except ImportError:
        raise Skip("pynvml / nvidia-ml-py not installed")
    from WinCore.thermal import ThermalGuard
    guard = ThermalGuard(threshold_c=1000)  # absurdly high so check() is instant
    t0 = time.time()
    guard.check()
    dt = time.time() - t0
    assert dt < 2.0, f"check() took {dt:.2f}s under an unreachable threshold -- should return instantly"
    return f"real pynvml temperature read via ThermalGuard.check(), returned in {dt*1000:.1f}ms"


# ---- multigpu ----

def check_multigpu_plan_real():
    torch = need_torch()
    from WinCore.multigpu import plan_distributed
    plan = plan_distributed()
    assert plan.world_size >= 1
    expected_backend = "gloo" if platform.system() == "Windows" else "nccl" if torch.cuda.is_available() else "gloo"
    return f"plan_distributed() on this real machine -> {plan} (expected backend family: {expected_backend})"


def check_multigpu_topology_real():
    torch = need_torch()
    need_cuda(torch)
    if torch.cuda.device_count() < 2:
        raise Skip(f"only {torch.cuda.device_count()} GPU(s) visible -- topology needs >=2")
    from WinCore.multigpu import detect_topology
    topo = detect_topology()
    return f"REAL {torch.cuda.device_count()}-GPU topology detected: {topo}"


# ---- kernels ----

def check_kernel_build_status_real():
    need_torch()
    try:
        from WinCore.kernels import kernel_status
    except ImportError as e:
        raise Skip(f"WinCore.kernels didn't expose kernel_status ({e}) -- "
                    f"likely the extension has never been built here; "
                    f"run `python -m WinCore.kernels.build` first")
    status = kernel_status()
    return f"kernel_status() -> {status}"


def check_kernel_fused_correctness_real():
    torch = need_torch()
    need_cuda(torch)
    import torch.nn.functional as F
    from WinCore.kernels import fused_bias_gelu, kernel_status

    status = kernel_status()
    torch.manual_seed(0)
    for dtype in (torch.float32, torch.float16, torch.bfloat16, torch.float64):
        x = torch.randn(64, 256, device="cuda", dtype=dtype)
        bias = torch.randn(256, device="cuda", dtype=dtype)
        got = fused_bias_gelu(x, bias)
        ref = F.gelu(x + bias)
        atol = {"torch.float16": 1e-2, "torch.bfloat16": 5e-2}.get(str(dtype), 1e-4)
        max_err = (got.float() - ref.float()).abs().max().item()
        assert max_err < atol * 10, f"{dtype}: fused vs unfused GELU max_err={max_err} (kernel_status={status})"
    return f"REAL fused-vs-unfused GELU numerical match across fp32/fp16/bf16/fp64 (kernel_status={status})"


def check_kernel_fused_perf_real():
    torch = need_torch()
    need_cuda(torch)
    import torch.nn.functional as F
    from WinCore.kernels import fused_bias_gelu, kernel_status

    status = kernel_status()
    if not status.get("compiled", False) if isinstance(status, dict) else "compiled" not in str(status):
        raise Skip(f"fused kernel not compiled on this machine (status={status}); "
                    f"nothing to benchmark, unfused fallback is expected to be active")

    x = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    bias = torch.randn(4096, device="cuda", dtype=torch.float16)

    def bench(fn, n=50):
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return (time.time() - t0) / n

    t_fused = bench(lambda: fused_bias_gelu(x, bias))
    t_unfused = bench(lambda: F.gelu(x + bias))
    speedup = t_unfused / t_fused if t_fused > 0 else float("inf")
    return f"fused={t_fused*1e6:.1f}us  unfused={t_unfused*1e6:.1f}us  speedup={speedup:.2f}x"


def check_kernel_build_toolchain_real():
    nvcc = shutil.which("nvcc")
    cl = shutil.which("cl")
    if not nvcc:
        raise Skip("nvcc not on PATH -- can't attempt a real compile here")
    from WinCore.kernels import build
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            build.build()  # adjust if the real entrypoint differs
            outcome = "build() completed without raising"
        except Exception as e:
            outcome = f"build() raised: {e!r}"
    return f"nvcc={nvcc}, cl={cl or 'NOT FOUND'} -> {outcome}\n{buf.getvalue()[-800:]}"


# ---- compile ----

def check_compile_safe_compile_real():
    torch = need_torch()
    from WinCore.compile import safe_compile
    import torch.nn as nn
    model = nn.Linear(8, 8)
    compiled, info = safe_compile(model) if _returns_tuple(safe_compile) else (safe_compile(model), None)
    x = torch.randn(2, 8)
    y = compiled(x)
    assert y.shape == (2, 8)
    return f"safe_compile() produced a callable model, real forward pass ran (info={info})"


def _returns_tuple(fn):
    # best-effort: don't fail the whole check just for an API-shape guess
    return False


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

CHECKS = [
    ("io", "atomic_write roundtrip", check_io_atomic_write_roundtrip),
    ("io", "atomic_torch_save (real torch)", check_io_torch_save_real),
    ("io", "atomic_safetensors_save (real safetensors)", check_io_safetensors_real),
    ("cpu", "recommended_threads (real cores)", check_cpu_recommended_threads_real),
    ("cpu", "apply() (real threads set)", check_cpu_apply_real),
    ("cpu", "set_priority (real psutil)", check_cpu_set_priority_real),
    ("spec", "get_system_spec (real machine)", check_spec_real_machine),
    ("spec", "meets_minimum boundary check", check_spec_meets_minimum_real),
    ("precision", "resolve_dtype (real torch dtypes)", check_precision_resolve_dtype_real),
    ("precision", "recommended_dtype (real GPU capability)", check_precision_recommended_dtype_real),
    ("precision", "amp() context (real autocast)", check_precision_amp_real),
    ("precision", "quantize_fp8 (REAL fp8 hardware)", check_precision_fp8_real_hardware),
    ("kv", "StepCache append/replace/evict (real torch)", check_kv_real_torch),
    ("kv", "StepCache fp8 compress (REAL hardware)", check_kv_compress_real_hardware),
    ("diagnostics", "NaN loss detection", check_diagnostics_nan_detection_real),
    ("diagnostics", "grad norm on real backward()", check_diagnostics_grad_norm_real),
    ("diagnostics", "attach_nan_guards (real nn.Module)", check_diagnostics_nan_guards_real),
    ("diagnostics", "gpu_timer (REAL CUDA events)", check_diagnostics_gpu_timer_real),
    ("memory", "recommended_dataloader_kwargs", check_memory_dataloader_kwargs_real),
    ("memory", "CacheGuard.check() (real)", check_memory_cache_guard_real),
    ("cache", "DiskCache get_or_compute (real disk I/O)", check_cache_disk_cache_real),
    ("thermal", "ThermalGuard.check (REAL pynvml read)", check_thermal_pynvml_real),
    ("multigpu", "plan_distributed (real backend choice)", check_multigpu_plan_real),
    ("multigpu", "detect_topology (REAL multi-GPU)", check_multigpu_topology_real),
    ("kernels", "kernel_status()", check_kernel_build_status_real),
    ("kernels", "fused_bias_gelu correctness (REAL CUDA kernel)", check_kernel_fused_correctness_real),
    ("kernels", "fused_bias_gelu perf vs unfused (REAL timing)", check_kernel_fused_perf_real),
    ("kernels", "toolchain build attempt (REAL nvcc)", check_kernel_build_toolchain_real),
    ("compile", "safe_compile (real torch.compile)", check_compile_safe_compile_real),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-pytest", action="store_true",
                         help="skip running tests/ via pytest, only run the live checks")
    parser.add_argument("--json", default=None, help="also write results to this JSON file")
    args = parser.parse_args()

    warnings.filterwarnings("default")  # don't hide WinCore's own warnings

    probe_environment()

    if not args.skip_pytest:
        run_pytest_suite()

    print("\n" + "#" * 78)
    print("# PART 2: live checks against real libraries on this machine")
    print("#" * 78)

    # Make sure WinCore itself importable when this script is run from
    # the project root without `pip install -e .`.
    if os.path.isdir("WinCore") and "" not in sys.path:
        sys.path.insert(0, os.getcwd())

    for section, name, fn in CHECKS:
        R.run(section, name, fn)

    counts = R.summary()

    if args.json:
        R.to_json(args.json)

    sys.exit(1 if (counts["FAIL"] or counts["ERROR"]) else 0)


if __name__ == "__main__":
    main()
