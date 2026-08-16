"""
Correctness test for WinCore.kernels.fused_bias_gelu.

Skips automatically if there's no CUDA GPU available -- this sandbox has
none, so this test has NOT actually been run yet. Run `pytest
tests/test_fused_bias_gelu.py -v` on a real CUDA machine (after
`python -m WinCore.kernels.build` succeeds) to get real pass/fail
results and, separately, real timing numbers vs. the unfused version.

Dtype coverage: float32/float64/float16/bfloat16 run the same
correctness checks as the original float32-only version, parametrized
-- these exercise the new templated kernel paths added to the .cu file
and are UNVERIFIED until run on real hardware, same as float32 was
before this file was first run. fp8 (e4m3/e5m2) is tested separately
against a looser tolerance since it goes through the upcast-to-float32
bridge (see fused_bias_gelu.py) rather than a native fp8 kernel path,
and is skipped on a torch build without float8 dtypes.
"""
import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA GPU (none available here)"
)

_DTYPES = [torch.float32, torch.float64, torch.float16, torch.bfloat16]
_DTYPE_IDS = ["fp32", "fp64", "fp16", "bf16"]

_FP8_DTYPES = [
    getattr(torch, name)
    for name in ("float8_e4m3fn", "float8_e5m2")
    if hasattr(torch, name)
]


def _reference(x, bias):
    import torch.nn.functional as F

    return F.gelu(x + bias, approximate="tanh")


@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
def test_forward_matches_unfused_reference(dtype):
    from WinCore.kernels import fused_bias_gelu

    torch.manual_seed(0)
    x = torch.randn(64, 128, device="cuda", dtype=dtype)
    bias = torch.randn(128, device="cuda", dtype=dtype)

    out = fused_bias_gelu(x, bias)
    ref = _reference(x, bias)

    # Wider tolerance for fp16/bf16 -- lower mantissa precision means
    # the fused and unfused paths can legitimately round differently
    # by a bit more than fp32/fp64 do.
    atol, rtol = (1e-5, 1e-4) if dtype in (torch.float32, torch.float64) else (1e-2, 1e-2)
    assert torch.allclose(out.float(), ref.float(), atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
def test_backward_matches_autograd_reference(dtype):
    from WinCore.kernels import fused_bias_gelu

    torch.manual_seed(0)
    x = torch.randn(32, 64, device="cuda", dtype=dtype, requires_grad=True)
    bias = torch.randn(64, device="cuda", dtype=dtype, requires_grad=True)

    out = fused_bias_gelu(x, bias)
    out.sum().backward()
    grad_x_fused, grad_bias_fused = x.grad.clone(), bias.grad.clone()

    x.grad = None
    bias.grad = None
    ref = _reference(x, bias)
    ref.sum().backward()

    atol, rtol = (1e-4, 1e-3) if dtype in (torch.float32, torch.float64) else (2e-2, 2e-2)
    assert torch.allclose(grad_x_fused.float(), x.grad.float(), atol=atol, rtol=rtol)
    assert torch.allclose(grad_bias_fused.float(), bias.grad.float(), atol=atol, rtol=rtol)


@pytest.mark.skipif(not _FP8_DTYPES, reason="this torch build has no float8 dtypes (need PyTorch 2.1+)")
@pytest.mark.parametrize("dtype", _FP8_DTYPES)
def test_fp8_forward_is_correct_via_upcast_bridge(dtype):
    """fp8 goes through the upcast-to-float32 bridge (see
    fused_bias_gelu.py), not a native fp8 kernel path -- so this checks
    correctness only, with an fp8-appropriate (loose) tolerance, and
    doesn't assert anything about speed."""
    from WinCore.kernels import fused_bias_gelu

    torch.manual_seed(0)
    x = torch.randn(64, 128, device="cuda").to(dtype)
    bias = torch.randn(128, device="cuda").to(dtype)

    out = fused_bias_gelu(x, bias)
    ref = _reference(x.float(), bias.float())

    assert torch.allclose(out.float(), ref, atol=0.2, rtol=0.2)


def test_fused_is_at_least_as_fast_as_unfused_for_large_tensor():
    """Sanity timing check -- fusion should win on a large elementwise
    fp32 tensor because it avoids one VRAM round-trip. This is a real
    timing assertion, not a fabricated number; if it fails on your GPU
    that's real information (e.g. very old GPU where kernel launch
    overhead dominates), not something to silently ignore. Only checked
    for fp32 -- other dtypes go through the same fused kernel mechanism,
    so this isn't re-asserted per-dtype."""
    import time
    from WinCore.kernels import fused_bias_gelu, kernel_status

    x = torch.randn(4096, 4096, device="cuda", dtype=torch.float32)
    bias = torch.randn(4096, device="cuda", dtype=torch.float32)

    torch.cuda.synchronize()
    for _ in range(3):  # warmup
        _reference(x, bias)
        fused_bias_gelu(x, bias)
    torch.cuda.synchronize()

    status = kernel_status()
    if status.backend != "cuda_extension":
        pytest.skip(
            "compiled CUDA kernel unavailable on this machine "
            f"({status.reason!r}) -- fused_bias_gelu is running on the "
            "PyTorch fallback, so a speedup assertion doesn't apply here."
        )

    t0 = time.perf_counter()
    for _ in range(50):
        _reference(x, bias)
    torch.cuda.synchronize()
    t_unfused = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(50):
        fused_bias_gelu(x, bias)
    torch.cuda.synchronize()
    t_fused = time.perf_counter() - t0

    print(f"\nunfused: {t_unfused:.4f}s  fused: {t_fused:.4f}s")
    assert t_fused < t_unfused
