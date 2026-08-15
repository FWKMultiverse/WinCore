import pytest

torch = pytest.importorskip("torch")

from WinCore.precision import amp, dequantize_fp8, quantize_fp8, resolve_dtype


def _has_fp8_cuda():
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability(0)[0] >= 8
        and hasattr(torch, "float8_e4m3fn")
        # compute capability >= 8.9 (Hopper/Ada+) is needed for real
        # fp8 hardware support; 8.0-8.6 (Ampere) has the dtype but not
        # the fast kernels -- quantize_fp8 will still *run* on Ampere
        # (it's a cast + elementwise op, not a fused fp8 kernel), so
        # this check is deliberately loose (>= 8) rather than requiring
        # exactly Hopper/Ada, matching what the sandbox can exercise.
    )


def test_quantize_fp8_roundtrip_preserves_magnitude():
    if not _has_fp8_cuda():
        pytest.skip("needs a CUDA device with float8 dtype support")
    x = torch.randn(64, 64, device="cuda", dtype=torch.float16) * 3.0
    packed = quantize_fp8(x, fmt="e4m3")
    restored = dequantize_fp8(packed)
    assert restored.dtype == x.dtype
    assert restored.shape == x.shape
    # fp8 is lossy by design -- assert the reconstruction is in the
    # right ballpark, not bit-exact.
    rel_err = (restored - x).abs().mean() / x.abs().mean()
    assert rel_err < 0.15


def test_quantize_fp8_zero_tensor_does_not_divide_by_zero():
    if not _has_fp8_cuda():
        pytest.skip("needs a CUDA device with float8 dtype support")
    x = torch.zeros(8, 8, device="cuda", dtype=torch.float16)
    packed = quantize_fp8(x)
    assert packed.scale == 1.0
    restored = dequantize_fp8(packed)
    assert torch.all(restored == 0)


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
