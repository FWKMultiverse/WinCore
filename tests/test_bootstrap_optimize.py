"""
Tests for WinCore.bootstrap.optimize() -- the one-call sequencing
wrapper around WinCore.cpu.apply() + WinCore.precision.cuda_perf_defaults().

Both underlying functions already have their own thorough test suites
(test_cpu.py, test_precision_cuda_perf.py) covering their real
mechanisms. These tests only check that optimize() calls them in the
right order, forwards kwargs correctly, and assembles the combined
OptimizePlan/warnings correctly -- so WinCore.cpu / WinCore.precision
are monkeypatched with simple fakes rather than re-testing their
internals here.
"""
from dataclasses import dataclass, field
from typing import Tuple

import pytest

import WinCore.bootstrap as bootstrap


@dataclass
class _FakeThreadPlan:
    total_logical: int = 16
    reserved: int = 2
    recommended: int = 14
    warnings: tuple = ()


@dataclass
class _FakeCudaPerfPlan:
    applied: bool = True
    tf32: bool = True
    cudnn_benchmark: bool = True
    sdpa_backends_enabled: list = field(default_factory=lambda: ["flash"])
    warnings: list = field(default_factory=list)
    reason: str = "fake"


def test_optimize_calls_cpu_then_cuda_in_order(monkeypatch):
    call_order = []

    def fake_apply(**kwargs):
        call_order.append("cpu")
        return _FakeThreadPlan()

    def fake_cuda_perf_defaults(**kwargs):
        call_order.append("cuda")
        return _FakeCudaPerfPlan()

    # bootstrap.optimize() does `from . import cpu as _cpu` INSIDE the
    # function body (lazy, same convention as everywhere else in this
    # package), so patch at the source modules instead of the
    # bootstrap module's own namespace.
    import WinCore.cpu as real_cpu_mod
    import WinCore.precision as real_precision_mod

    monkeypatch.setattr(real_cpu_mod, "apply", fake_apply)
    monkeypatch.setattr(real_precision_mod, "cuda_perf_defaults", fake_cuda_perf_defaults)

    plan = bootstrap.optimize()
    assert call_order == ["cpu", "cuda"]
    assert isinstance(plan.cpu, _FakeThreadPlan)
    assert isinstance(plan.cuda, _FakeCudaPerfPlan)


def test_optimize_forwards_cpu_kwargs(monkeypatch):
    seen_kwargs = {}

    def fake_apply(**kwargs):
        seen_kwargs.update(kwargs)
        return _FakeThreadPlan()

    import WinCore.cpu as real_cpu_mod
    import WinCore.precision as real_precision_mod

    monkeypatch.setattr(real_cpu_mod, "apply", fake_apply)
    monkeypatch.setattr(real_precision_mod, "cuda_perf_defaults", lambda **kwargs: _FakeCudaPerfPlan())

    bootstrap.optimize(cpu_kwargs={"priority": "above_normal"})
    assert seen_kwargs == {"priority": "above_normal"}


def test_optimize_forwards_cuda_kwargs(monkeypatch):
    seen_kwargs = {}

    def fake_cuda(**kwargs):
        seen_kwargs.update(kwargs)
        return _FakeCudaPerfPlan()

    import WinCore.cpu as real_cpu_mod
    import WinCore.precision as real_precision_mod

    monkeypatch.setattr(real_cpu_mod, "apply", lambda **kwargs: _FakeThreadPlan())
    monkeypatch.setattr(real_precision_mod, "cuda_perf_defaults", fake_cuda)

    bootstrap.optimize(cuda_kwargs={"cudnn_benchmark": False})
    assert seen_kwargs == {"cudnn_benchmark": False}


def test_optimize_apply_cuda_false_skips_cuda_entirely(monkeypatch):
    cuda_called = []

    import WinCore.cpu as real_cpu_mod
    import WinCore.precision as real_precision_mod

    monkeypatch.setattr(real_cpu_mod, "apply", lambda **kwargs: _FakeThreadPlan())
    monkeypatch.setattr(
        real_precision_mod,
        "cuda_perf_defaults",
        lambda **kwargs: (cuda_called.append(True), _FakeCudaPerfPlan())[1],
    )

    plan = bootstrap.optimize(apply_cuda=False)
    assert cuda_called == []
    assert plan.cuda is None


def test_optimize_flattens_and_prefixes_warnings(monkeypatch):
    import WinCore.cpu as real_cpu_mod
    import WinCore.precision as real_precision_mod

    monkeypatch.setattr(
        real_cpu_mod, "apply", lambda **kwargs: _FakeThreadPlan(warnings=("psutil missing",))
    )
    monkeypatch.setattr(
        real_precision_mod,
        "cuda_perf_defaults",
        lambda **kwargs: _FakeCudaPerfPlan(warnings=["tf32 not capable"]),
    )

    plan = bootstrap.optimize()
    assert plan.warnings == ["cpu: psutil missing", "cuda: tf32 not capable"]


def test_optimize_no_warnings_gives_empty_list(monkeypatch):
    import WinCore.cpu as real_cpu_mod
    import WinCore.precision as real_precision_mod

    monkeypatch.setattr(real_cpu_mod, "apply", lambda **kwargs: _FakeThreadPlan())
    monkeypatch.setattr(real_precision_mod, "cuda_perf_defaults", lambda **kwargs: _FakeCudaPerfPlan())

    plan = bootstrap.optimize()
    assert plan.warnings == []
