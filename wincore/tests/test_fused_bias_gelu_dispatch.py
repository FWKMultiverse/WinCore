"""
Tests for the overhead-aware size threshold in
WinCore.kernels.fused_bias_gelu -- the logic that decides whether a
given tensor is worth routing through the compiled CUDA extension at
all, vs. just running the plain-PyTorch path directly (see the
"Overhead-aware dispatch" section of that module's docstring for why
this exists -- it's a direct response to the first real benchmark
showing the fused kernel LOSING to the unfused path on a small tensor).

This only needs `torch` to be importable (fused_bias_gelu.py imports
it at module level) -- it does NOT need a CUDA GPU, since
`_should_use_fused_kernel` and the threshold-resolution functions are
plain Python integer/string logic with no tensor math in them. Skips
cleanly if torch itself isn't installed at all.
"""
import os

import pytest

torch = pytest.importorskip("torch")

from WinCore.kernels.fused_bias_gelu import (
    _should_use_fused_kernel,
    _min_elements_for_fusion,
    current_fusion_threshold,
    _DEFAULT_MIN_ELEMENTS_FOR_FUSION,
)


def test_should_use_fused_kernel_below_threshold_is_false():
    assert _should_use_fused_kernel(numel=100, min_elements=1000) is False


def test_should_use_fused_kernel_at_threshold_is_true():
    assert _should_use_fused_kernel(numel=1000, min_elements=1000) is True


def test_should_use_fused_kernel_just_below_threshold_is_false():
    assert _should_use_fused_kernel(numel=999, min_elements=1000) is False


def test_should_use_fused_kernel_well_above_threshold_is_true():
    assert _should_use_fused_kernel(numel=50_000_000, min_elements=1_048_576) is True


def test_default_threshold_used_when_no_env_var_and_not_calibrated(monkeypatch):
    monkeypatch.delenv("WINCORE_FUSED_MIN_ELEMENTS", raising=False)
    import importlib; mod = importlib.import_module("WinCore.kernels.fused_bias_gelu")  # NOT "import ... as mod" -- WinCore/kernels/__init__.py deliberately re-exports the fused_bias_gelu FUNCTION at package level, which shadows the submodule for dotted "as" imports

    monkeypatch.setattr(mod, "_calibrated_min_elements", None)
    assert _min_elements_for_fusion() == _DEFAULT_MIN_ELEMENTS_FOR_FUSION
    assert current_fusion_threshold() == _DEFAULT_MIN_ELEMENTS_FOR_FUSION


def test_env_var_overrides_default(monkeypatch):
    import importlib; mod = importlib.import_module("WinCore.kernels.fused_bias_gelu")  # NOT "import ... as mod" -- WinCore/kernels/__init__.py deliberately re-exports the fused_bias_gelu FUNCTION at package level, which shadows the submodule for dotted "as" imports

    monkeypatch.setattr(mod, "_calibrated_min_elements", None)
    monkeypatch.setenv("WINCORE_FUSED_MIN_ELEMENTS", "2048")
    assert _min_elements_for_fusion() == 2048


def test_invalid_env_var_falls_back_to_default_with_warning(monkeypatch):
    import importlib; mod = importlib.import_module("WinCore.kernels.fused_bias_gelu")  # NOT "import ... as mod" -- WinCore/kernels/__init__.py deliberately re-exports the fused_bias_gelu FUNCTION at package level, which shadows the submodule for dotted "as" imports

    monkeypatch.setattr(mod, "_calibrated_min_elements", None)
    monkeypatch.setenv("WINCORE_FUSED_MIN_ELEMENTS", "not-a-number")
    with pytest.warns(RuntimeWarning):
        result = _min_elements_for_fusion()
    assert result == _DEFAULT_MIN_ELEMENTS_FOR_FUSION


def test_calibrated_value_takes_priority_over_env_var(monkeypatch):
    import importlib; mod = importlib.import_module("WinCore.kernels.fused_bias_gelu")  # NOT "import ... as mod" -- WinCore/kernels/__init__.py deliberately re-exports the fused_bias_gelu FUNCTION at package level, which shadows the submodule for dotted "as" imports

    monkeypatch.setenv("WINCORE_FUSED_MIN_ELEMENTS", "2048")
    monkeypatch.setattr(mod, "_calibrated_min_elements", 99999)
    assert _min_elements_for_fusion() == 99999
