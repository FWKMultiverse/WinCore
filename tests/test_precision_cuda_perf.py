"""
Tests for WinCore.precision.cuda_perf_defaults() -- see that function's
docstring for what each torch.backends knob actually does. This is
tested against a fully mocked `torch` module (via sys.modules,
monkeypatched) rather than the real `_fake_torch.py` shim, because it
needs `torch.backends.cuda`/`torch.backends.cudnn` submodule structure
that shim doesn't model -- same monkeypatch-sys.modules convention
already used in test_io_checkpoint_formats.py / test_io_fast_load.py
for the same "don't require torch installed to test the dispatch
logic" reason.
"""
import sys
import types

import pytest

from WinCore.precision import cuda_perf_defaults


def _make_fake_torch(cuda_available=True, capability=(8, 0), has_sdpa_toggles=True, has_set_precision=True):
    mod = types.ModuleType("torch")

    class _Cuda:
        @staticmethod
        def is_available():
            return cuda_available

        @staticmethod
        def get_device_capability(index=0):
            return capability

    class _Matmul:
        allow_tf32 = False

    class _BackendsCuda:
        matmul = _Matmul()
        _sdp_calls = {}

        def enable_flash_sdp(self, v):
            self._sdp_calls["flash"] = v

        def enable_mem_efficient_sdp(self, v):
            self._sdp_calls["mem_efficient"] = v

        def enable_math_sdp(self, v):
            self._sdp_calls["math"] = v

    class _Cudnn:
        allow_tf32 = False
        benchmark = False

    class _Backends:
        cuda = _BackendsCuda()
        cudnn = _Cudnn()

    if not has_sdpa_toggles:
        del _BackendsCuda.enable_flash_sdp
        del _BackendsCuda.enable_mem_efficient_sdp
        del _BackendsCuda.enable_math_sdp

    mod.cuda = _Cuda()
    mod.backends = _Backends()
    if has_set_precision:
        mod.set_float32_matmul_precision = lambda mode: setattr(mod, "_matmul_precision", mode)
    return mod


def test_no_cuda_returns_inert_plan(monkeypatch):
    fake = _make_fake_torch(cuda_available=False)
    monkeypatch.setitem(sys.modules, "torch", fake)

    plan = cuda_perf_defaults()
    assert plan.applied is False
    assert plan.tf32 is False
    assert plan.cudnn_benchmark is False
    assert plan.sdpa_backends_enabled == []
    assert "No CUDA" in plan.reason


def test_ampere_plus_enables_tf32(monkeypatch):
    fake = _make_fake_torch(cuda_available=True, capability=(8, 0))
    monkeypatch.setitem(sys.modules, "torch", fake)

    plan = cuda_perf_defaults()
    assert plan.tf32 is True
    assert fake.backends.cuda.matmul.allow_tf32 is True
    assert fake.backends.cudnn.allow_tf32 is True
    assert fake._matmul_precision == "high"


def test_pre_ampere_leaves_tf32_off_and_warns(monkeypatch):
    fake = _make_fake_torch(cuda_available=True, capability=(7, 5))  # Turing
    monkeypatch.setitem(sys.modules, "torch", fake)

    plan = cuda_perf_defaults()
    assert plan.tf32 is False
    assert fake.backends.cuda.matmul.allow_tf32 is False
    assert fake.backends.cudnn.allow_tf32 is False
    assert fake._matmul_precision == "highest"
    assert any("compute capability" in w for w in plan.warnings)


def test_cudnn_benchmark_flag_respected(monkeypatch):
    fake = _make_fake_torch(cuda_available=True)
    monkeypatch.setitem(sys.modules, "torch", fake)

    plan = cuda_perf_defaults(cudnn_benchmark=False)
    assert plan.cudnn_benchmark is False
    assert fake.backends.cudnn.benchmark is False

    fake2 = _make_fake_torch(cuda_available=True)
    monkeypatch.setitem(sys.modules, "torch", fake2)
    plan2 = cuda_perf_defaults(cudnn_benchmark=True)
    assert plan2.cudnn_benchmark is True
    assert fake2.backends.cudnn.benchmark is True


def test_sdpa_backends_all_enabled_by_default(monkeypatch):
    fake = _make_fake_torch(cuda_available=True)
    monkeypatch.setitem(sys.modules, "torch", fake)

    plan = cuda_perf_defaults()
    assert set(plan.sdpa_backends_enabled) == {"flash", "mem_efficient", "math"}
    assert fake.backends.cuda._sdp_calls == {"flash": True, "mem_efficient": True, "math": True}


def test_sdpa_backends_skipped_when_requested(monkeypatch):
    fake = _make_fake_torch(cuda_available=True)
    monkeypatch.setitem(sys.modules, "torch", fake)

    plan = cuda_perf_defaults(sdpa_backends=False)
    assert plan.sdpa_backends_enabled == []
    assert fake.backends.cuda._sdp_calls == {}


def test_missing_sdpa_toggles_on_old_torch_warns_but_does_not_raise(monkeypatch):
    fake = _make_fake_torch(cuda_available=True, has_sdpa_toggles=False)
    monkeypatch.setitem(sys.modules, "torch", fake)

    plan = cuda_perf_defaults()
    assert plan.sdpa_backends_enabled == []
    assert any("not present on this torch build" in w for w in plan.warnings)


def test_apply_false_reports_plan_without_mutating_backends(monkeypatch):
    fake = _make_fake_torch(cuda_available=True, capability=(8, 0))
    monkeypatch.setitem(sys.modules, "torch", fake)

    plan = cuda_perf_defaults(apply=False)
    assert plan.applied is False
    assert plan.tf32 is True  # what WOULD be applied
    # backends untouched -- still at the fake's initial defaults
    assert fake.backends.cuda.matmul.allow_tf32 is False
    assert fake.backends.cudnn.benchmark is False
    assert fake.backends.cuda._sdp_calls == {}


def test_missing_set_float32_matmul_precision_is_tolerated(monkeypatch):
    fake = _make_fake_torch(cuda_available=True, has_set_precision=False)
    monkeypatch.setitem(sys.modules, "torch", fake)

    plan = cuda_perf_defaults()  # must not raise
    assert plan.tf32 is True
