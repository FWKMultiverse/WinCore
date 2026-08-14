import pytest

torch = pytest.importorskip("torch")

from WinCore.precision import amp, resolve_dtype


def test_resolve_common_aliases():
    assert resolve_dtype("fp16") == torch.float16
    assert resolve_dtype("bf16") == torch.bfloat16
    assert resolve_dtype("fp32") == torch.float32
    assert resolve_dtype("fp64") == torch.float64


def test_resolve_rejects_fp4():
    with pytest.raises(ValueError):
        resolve_dtype("fp4")


def test_resolve_rejects_unknown():
    with pytest.raises(ValueError):
        resolve_dtype("not_a_dtype")


def test_amp_disabled_without_cuda():
    """No GPU in this environment (this sandbox has neither a GPU nor
    torch installed with CUDA support to test against) -- this only
    checks the CPU-only fallback path is inert, not the CUDA path."""
    if torch.cuda.is_available():
        pytest.skip("this test targets the no-CUDA fallback path specifically")
    ctx = amp()
    assert ctx.plan.enabled is False
    assert ctx.scaler.is_enabled() is False
    with ctx.autocast():
        pass  # must not raise, must not actually do anything


def test_amp_uses_grad_scaler_only_for_fp16():
    """Real assertion on the resolution rule (fp16 -> scaler on, bf16 ->
    scaler off); the CUDA-dtype branch itself has not been run on real
    hardware in this sandbox (no GPU here) -- verify on a CUDA machine
    before relying on it."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device to exercise the enabled path")
    ctx = amp()
    if ctx.plan.dtype is torch.float16:
        assert ctx.plan.use_grad_scaler is True
        assert ctx.scaler.is_enabled() is True
    elif ctx.plan.dtype is torch.bfloat16:
        assert ctx.plan.use_grad_scaler is False
        assert ctx.scaler.is_enabled() is False
