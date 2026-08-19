"""
Tests for the item-4 precision additions:
  - quantize_fp8(..., axis=...) per-channel scaling
  - quantize_fp4/dequantize_fp4 (experimental 4-bit) + the pure
    _pack_4bit_codes/_unpack_4bit_codes bit-math helpers
  - safe_cast() fp16/fp8 overflow handling

Runs against tests/_fake_torch.py (auto-installed by conftest.py when
real torch isn't available) -- see test_precision.py's own comment on
why `import torch` (not importorskip-and-skip) is what actually
exercises this in the sandbox.
"""
import warnings

import pytest

torch = pytest.importorskip("torch")

from WinCore.precision import (
    Fp4Tensor,
    dequantize_fp4,
    dequantize_fp8,
    quantize_fp4,
    quantize_fp8,
    safe_cast,
    _pack_4bit_codes,
    _unpack_4bit_codes,
)


def _approx_equal(a, b, rel=1e-6, abs_tol=1e-6):
    """pytest.approx isn't in the offline test-runner's pytest shim --
    this is a minimal stand-in, used only in this file. Handles plain
    floats and (possibly nested) lists of floats element-wise."""
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_approx_equal(x, y, rel, abs_tol) for x, y in zip(a, b))
    return abs(a - b) <= max(abs_tol, rel * max(abs(a), abs(b)))


# -- per-axis fp8 quantization -------------------------------------------


def test_quantize_fp8_axis_none_matches_previous_global_behavior():
    t = torch.tensor([[100.0, 50.0], [1.0, -2.0]])
    q = quantize_fp8(t, fmt="e4m3")
    assert q.axis is None
    assert isinstance(q.scale, float)
    restored = dequantize_fp8(q)
    # BUGFIX (found via a real-hardware test run): e4m3 has only ~3
    # mantissa bits, so real fp8 rounding is genuinely lossy -- up to
    # roughly a 6-12% relative error per element is normal, expected
    # behavior, not a bug (see this module's own docstring, and the
    # already-hardware-validated test_quantize_fp8_roundtrip_preserves_magnitude
    # in test_precision.py, which uses the same 15% bound this test now
    # matches). The original abs_tol=1e-6 here only ever passed because
    # the sandbox's fake torch shim doesn't perform real bit-level fp8
    # rounding at all (a pure bookkeeping no-op) -- it silently
    # encoded an incorrect "fp8 round-trips losslessly" assumption
    # that could only ever be caught by testing on real hardware.
    assert _approx_equal(restored.tolist(), t.tolist(), rel=0.15, abs_tol=1e-3)


def test_quantize_fp8_per_axis_gives_distinct_scale_per_row():
    # row 0 has a much larger range than row 1 -- a single global
    # scale would force row 1's small values toward the noise floor.
    t = torch.tensor([[1000.0, -500.0], [1.0, -2.0]])
    q = quantize_fp8(t, fmt="e4m3", axis=0)
    assert q.axis == 0
    scales = q.scale.tolist()
    # two independent per-row scales, and they should differ a lot
    # given how different the two rows' magnitudes are
    assert len(scales) == 2
    assert scales[0][0] > scales[1][0] * 10


def test_quantize_fp8_per_axis_roundtrips_correctly():
    t = torch.tensor([[1000.0, -500.0, 250.0], [1.0, -2.0, 0.5]])
    q = quantize_fp8(t, fmt="e4m3", axis=0)
    restored = dequantize_fp8(q)
    # Same real-hardware-lossiness reasoning as
    # test_quantize_fp8_axis_none_matches_previous_global_behavior
    # above -- e4m3's ~3 mantissa bits mean a genuine, expected several-
    # percent relative error per element, not the near-exact round-trip
    # the sandbox's fake torch shim (which doesn't perform real bit-
    # level rounding) made this assertion look correct with.
    assert _approx_equal(restored.tolist(), t.tolist(), rel=0.15, abs_tol=1e-3)


def test_quantize_fp8_per_axis_all_zero_row_does_not_produce_nan():
    t = torch.tensor([[1.0, 2.0], [0.0, 0.0]])
    q = quantize_fp8(t, fmt="e4m3", axis=0)
    restored = dequantize_fp8(q)
    flat = restored.tolist()
    assert not any(v != v for row in flat for v in row)  # no NaN anywhere (v != v is the NaN check)
    assert flat[1] == [0.0, 0.0]


def test_quantize_fp8_negative_axis_normalized():
    t = torch.tensor([[1000.0, -500.0], [1.0, -2.0]])
    q_pos = quantize_fp8(t, fmt="e4m3", axis=0)
    q_neg = quantize_fp8(t, fmt="e4m3", axis=-2)  # same as axis=0 for a 2D tensor
    assert q_neg.axis == q_pos.axis == 0


def test_dequantize_fp8_handles_both_scalar_and_tensor_scale_uniformly():
    t = torch.tensor([[10.0, 20.0], [1.0, 2.0]])
    q_global = quantize_fp8(t, fmt="e4m3")
    q_axis = quantize_fp8(t, fmt="e4m3", axis=0)
    # both should dequantize without needing different calling code
    r1 = dequantize_fp8(q_global)
    r2 = dequantize_fp8(q_axis)
    assert r1.shape == r2.shape == t.shape


def test_quantize_fp8_empty_tensor_raises_clear_error():
    t = torch.tensor([])
    with pytest.raises(ValueError, match="empty tensor"):
        quantize_fp8(t)


def test_quantize_fp8_out_of_range_axis_raises():
    t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    for bad_axis in (2, -3, 5, -5):
        with pytest.raises(ValueError, match="out of range"):
            quantize_fp8(t, axis=bad_axis)


def test_quantize_fp8_negative_axis_within_range_is_accepted():
    t = torch.tensor([[1000.0, -500.0], [1.0, -2.0]])
    q_last = quantize_fp8(t, axis=-1)
    q_first = quantize_fp8(t, axis=-2)
    assert q_last.axis == 1
    assert q_first.axis == 0


# -- fp4 pure bit-math (no torch needed at all) --------------------------


def test_pack_unpack_roundtrip_even_count():
    codes = [0, 15, 7, 8, 1, 14]
    packed = _pack_4bit_codes(codes)
    assert len(packed) == 3
    assert _unpack_4bit_codes(packed, len(codes)) == codes


def test_pack_unpack_roundtrip_odd_count():
    codes = [3, 12, 9]
    packed = _pack_4bit_codes(codes)
    assert len(packed) == 2  # padded to 4 codes internally
    assert _unpack_4bit_codes(packed, len(codes)) == codes  # padding dropped correctly


def test_pack_known_values():
    # high nibble 0xF, low nibble 0x0 -> byte 0xF0 = 240
    assert _pack_4bit_codes([15, 0]) == [240]
    # high nibble 0x0, low nibble 0xF -> byte 0x0F = 15
    assert _pack_4bit_codes([0, 15]) == [15]
    # high 0xA, low 0x5 -> 0xA5 = 165
    assert _pack_4bit_codes([10, 5]) == [165]


def test_pack_masks_out_of_range_bits():
    # values outside 0-15 get masked to their low 4 bits (defensive,
    # not expected to happen from quantize_fp4's own callers, but the
    # pure function itself should not silently corrupt neighboring bits)
    assert _pack_4bit_codes([16 + 5, 3]) == _pack_4bit_codes([5, 3])


def test_unpack_empty_packed_list():
    assert _unpack_4bit_codes([], 0) == []


# -- fp4 quantize/dequantize (torch-facing) -------------------------------


def test_quantize_fp4_roundtrip_approximate():
    t = torch.tensor([1.0, -7.0, 3.5, 0.0, 7.0])
    q = quantize_fp4(t)
    assert isinstance(q, Fp4Tensor)
    assert q.numel == 5
    restored = dequantize_fp4(q)
    assert restored.shape == t.shape
    # 15-level quantization is lossy but should be in the right ballpark
    for orig, got in zip(t.tolist(), restored.tolist()):
        assert abs(orig - got) <= abs(orig) * 0.3 + 0.5


def test_quantize_fp4_preserves_shape_and_dtype():
    t = torch.tensor([[1.0, 2.0], [3.0, -4.0]])
    q = quantize_fp4(t)
    restored = dequantize_fp4(q)
    assert restored.shape == t.shape
    assert q.orig_shape == (2, 2)


def test_quantize_fp4_packed_is_half_the_element_count():
    t = torch.tensor([1.0, 2.0, 3.0, 4.0])  # even count, no padding needed
    q = quantize_fp4(t)
    assert len(q.packed.tolist()) == 2  # 4 codes -> 2 bytes


def test_quantize_fp4_odd_element_count_still_roundtrips():
    t = torch.tensor([1.0, 2.0, 3.0])  # odd count -> internal padding
    q = quantize_fp4(t)
    assert q.numel == 3
    restored = dequantize_fp4(q)
    assert restored.shape == t.shape
    assert len(restored.tolist()) == 3  # padding code never leaks into output


def test_quantize_fp4_all_zero_tensor_has_scale_one_and_zero_output():
    t = torch.tensor([0.0, 0.0, 0.0, 0.0])
    q = quantize_fp4(t)
    assert q.scale == 1.0
    restored = dequantize_fp4(q)
    assert restored.tolist() == [0.0, 0.0, 0.0, 0.0]


def test_quantize_fp4_max_value_maps_to_top_level():
    t = torch.tensor([10.0, -10.0])
    q = quantize_fp4(t)
    restored = dequantize_fp4(q)
    # both values are at the extreme -- should reconstruct to
    # something very close to +/-10 (level +/-7 * scale, scale=10/7)
    assert _approx_equal(restored.tolist()[0], 10.0, abs_tol=1e-6)
    assert _approx_equal(restored.tolist()[1], -10.0, abs_tol=1e-6)


def test_quantize_fp4_empty_tensor_raises_clear_error():
    t = torch.tensor([])
    with pytest.raises(ValueError, match="empty tensor"):
        quantize_fp4(t)


# -- safe_cast ---------------------------------------------------------


def test_safe_cast_in_range_value_no_warning():
    t = torch.tensor([100.0, -200.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning here fails the test
        result = safe_cast(t, torch.float16)
    assert result.dtype == torch.float16


def test_safe_cast_out_of_range_warns_by_default():
    t = torch.tensor([100000.0])  # exceeds float16's ~65504 max
    with pytest.warns(RuntimeWarning, match="exceeds"):
        result = safe_cast(t, torch.float16)
    assert result.dtype == torch.float16  # still casts, per "warn" semantics


def test_safe_cast_raise_mode_raises_overflow_error():
    t = torch.tensor([100000.0])
    with pytest.raises(OverflowError, match="exceeds"):
        safe_cast(t, torch.float16, on_overflow="raise")


def test_safe_cast_clip_mode_clamps_before_casting():
    t = torch.tensor([100000.0, -100000.0, 10.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = safe_cast(t, torch.float16, on_overflow="clip")
    values = result.tolist()
    assert _approx_equal(values[0], 65504.0, rel=1e-6)
    assert _approx_equal(values[1], -65504.0, rel=1e-6)
    assert _approx_equal(values[2], 10.0, abs_tol=1e-6)  # in-range value untouched


def test_safe_cast_skips_check_for_dtypes_without_known_overflow_risk():
    t = torch.tensor([1e30])  # would overflow fp16, NOT bfloat16 (same exponent range as fp32)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = safe_cast(t, torch.bfloat16)  # must not warn/raise
    assert result.dtype == torch.bfloat16


def test_safe_cast_checks_fp8_e4m3_range_too():
    t = torch.tensor([1000.0])  # exceeds e4m3's 448 max
    with pytest.warns(RuntimeWarning, match="exceeds"):
        safe_cast(t, torch.float8_e4m3fn)
