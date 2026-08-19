"""
Precision helper — pick a safe torch dtype instead of guessing, plus
real compression/safety helpers for fp8/fp4/fp16 storage.

Why this exists
----------------
`bfloat16` is only fast (or supported at all) on GPUs with compute
capability >= 8.0 (NVIDIA Ampere / RTX 30-series and newer). Using it on
older cards silently falls back to slow emulation or errors depending on
the op. `float16` works further back but can overflow in some training
loops (see `safe_cast()` below for a direct answer to that). This module
answers one narrow question honestly: "given the GPU that's actually in
this machine, which dtype should I default to?" — it does not claim to
support fp4 as a native compute dtype (no such IEEE/CUDA-native type
exists broadly; `quantize_fp4()`/`dequantize_fp4()` below are an
explicitly experimental, software-only, non-hardware-accelerated
*storage* compression scheme, not a compute dtype — see that function's
own docstring for exactly what that does and doesn't mean).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


def recommended_dtype(device_index: int = 0):
    """Return the best default torch dtype for the current CUDA device.

    Resolution:
      - No CUDA available -> `torch.float32` (CPU).
      - Compute capability >= 8.0 (Ampere+) -> `torch.bfloat16`
        (better dynamic range than fp16, native hardware support).
      - Compute capability >= 6.0 (Pascal+, e.g. GTX 10-series) ->
        `torch.float16`.
      - Older than that -> `torch.float32` (fp16 tensor cores aren't
        present; fp16 compute would be emulated and often slower).

    Import of torch is local so this module stays importable without
    torch installed; call this only when torch is available.
    """
    import torch

    if not torch.cuda.is_available():
        return torch.float32

    major, _minor = torch.cuda.get_device_capability(device_index)
    if major >= 8:
        return torch.bfloat16
    if major >= 6:
        return torch.float16
    return torch.float32


def dtype_name(dtype) -> str:
    return str(dtype).replace("torch.", "")


SUPPORTED_DTYPES = ("float32", "float64", "float16", "bfloat16", "float8_e4m3fn", "float8_e5m2")


def resolve_dtype(name: str):
    """Resolve a string ('fp16', 'bf16', 'fp32', 'fp64', 'fp8'/'fp8_e4m3'/
    'fp8_e5m2', or full torch names) to a torch dtype. Raises ValueError
    for anything unsupported — including 'fp4', which is intentionally
    not offered here (see module docstring).

    fp8 is real hardware-native storage on Hopper/Ada+ GPUs (compute
    capability >= 8.9) via `torch.float8_e4m3fn` / `torch.float8_e5m2`,
    added in PyTorch 2.1+ -- unlike fp4, so it's included here, but only
    on a torch build that actually has those attributes; on an older
    torch this raises a clear ValueError rather than a confusing
    AttributeError.
    """
    import torch

    aliases = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp64": torch.float64,
        "float64": torch.float64,
    }
    for key_name, attr in (
        ("fp8", "float8_e4m3fn"),
        ("fp8_e4m3", "float8_e4m3fn"),
        ("float8_e4m3fn", "float8_e4m3fn"),
        ("fp8_e5m2", "float8_e5m2"),
        ("float8_e5m2", "float8_e5m2"),
    ):
        if hasattr(torch, attr):
            aliases[key_name] = getattr(torch, attr)

    key = name.strip().lower()
    if key not in aliases:
        if key.startswith("fp8") or key.startswith("float8"):
            raise ValueError(
                f"'{name}' needs a torch build with float8 dtypes "
                "(PyTorch 2.1+); this installed torch doesn't have them. "
                f"Supported here: {sorted(set(aliases))}."
            )
        raise ValueError(
            f"Unsupported dtype '{name}'. Supported: {sorted(set(aliases))}. "
            f"(4-bit is a quantization scheme, not a compute dtype — use a "
            f"dedicated quantization library such as bitsandbytes for that.)"
        )
    return aliases[key]


@dataclass
class Fp8Tensor:
    """Result of `quantize_fp8()`: the fp8-storage tensor plus the
    scale(s) needed to bring it back to the original range. fp8
    (e4m3/e5m2) has only ~2-3 bits of mantissa, so unlike fp16/bf16
    it cannot store an arbitrary-magnitude tensor directly without
    clipping or flushing small values to zero -- a *scale* factor is
    required to fit the tensor's actual value range into fp8's narrow
    representable range before casting down, and to undo that scaling
    on the way back up. Storing `data` and `scale` together is what
    makes this a lossy-but-controlled compression scheme rather than
    "just cast to a smaller dtype and hope".

    `scale` is a plain Python `float` for the default (`axis=None`,
    whole-tensor) quantization, or a broadcastable `torch.Tensor` (one
    value per position along `axis`, size-1 elsewhere) when
    `axis` was given to `quantize_fp8()` -- see that function's
    docstring for why per-axis scaling exists and when it's worth the
    extra bookkeeping. `dequantize_fp8()` handles both forms
    identically (`packed.data.float() * packed.scale` broadcasts
    correctly either way), so calling code doesn't need to branch on
    which form `scale` is in."""

    data: "object"  # torch.Tensor, dtype=float8_e4m3fn or float8_e5m2
    scale: "object"  # float, or a broadcastable torch.Tensor if axis was given
    orig_dtype: "object"  # the dtype to restore on dequantize
    axis: Optional[int] = None  # which axis scale is per-position-of, if any (informational)


def quantize_fp8(tensor, fmt: str = "e4m3", axis: Optional[int] = None) -> Fp8Tensor:
    """Compress a float tensor to fp8 storage with dynamic per-tensor
    scaling, for cases where the goal is *storage/bandwidth* reduction
    (e.g. a KV cache, activation checkpointing, or an optimizer-state
    buffer) rather than autocast compute -- `amp()` above answers "what
    dtype should this GPU compute in", this answers "how do I actually
    shrink this tensor's memory footprint without just truncating bits
    and silently losing whatever didn't fit".

    Only requires a torch build with float8 dtypes (2.1+) -- same
    requirement as `resolve_dtype("fp8")`, whose clear `ValueError` this
    raises verbatim if that's missing, rather than a confusing
    AttributeError or silent wrong-dtype fallback. Despite an earlier
    version of this docstring, this function does NOT check for a
    Hopper/Ada+ (compute capability >= 8.9) GPU, and never has: the cast
    to float8_e4m3fn/float8_e5m2 here is a storage-only op (round + pack
    bits), which torch supports on any device -- CPU included -- once
    the dtype itself exists in the build. Compute capability >= 8.9 only
    matters if you then run *fused fp8 arithmetic kernels* against the
    result (see `WinCore.kernels.fused_bias_gelu`'s fp8 upcast-bridge
    note for why raw fp8 elementwise math isn't portable); this function
    stops at storage/bandwidth compression, so that requirement doesn't
    apply to it. If you need fp8 to also be numerically meaningful (not
    just a smaller footprint), check compute capability yourself before
    relying on the *quality* of the reconstruction; `quantize_fp8` will
    still run and return a valid (if info you already lost) `Fp8Tensor`
    on older/CPU-only hardware rather than refuse to.

    Args:
        tensor: a float16/bfloat16/float32 torch.Tensor.
        fmt: "e4m3" (default -- more mantissa bits, better for weights
            and activations with a moderate dynamic range) or "e5m2"
            (more exponent bits, better for gradients / values with a
            wide dynamic range, at the cost of precision).
        axis: `None` (default) computes ONE scale for the whole
            tensor, same as before this parameter existed. Pass an
            axis index (e.g. `axis=0` for a weight matrix shaped
            `(out_features, in_features)`) to compute one INDEPENDENT
            scale per position along that axis instead -- real
            precision improvement when different rows/channels have
            genuinely different magnitude distributions, which a
            single global scale forces into one compromise: rows with
            a much smaller range than the tensor's overall max get
            crushed toward zero (most of their fp8 codes go unused),
            while the single global scale is still set by whichever
            row/channel has the largest values. Per-axis scaling gives
            every position along `axis` its own fp8_max-relative
            headroom instead. Costs proportionally more scale storage
            (one float per position along `axis` instead of one for
            the whole tensor) -- for a `(4096, 4096)` weight matrix
            with `axis=0`, that's 4096 floats (16KB at fp32) against a
            multi-megabyte tensor, a small overhead for the precision
            gained. An all-zero row/channel along `axis` is handled
            safely (scale computed as if it were a tiny positive
            number instead of exactly zero, so that position's
            reconstruction is exactly zero, not NaN from a 0/0) --
            this per-axis path does not separately re-check the whole
            tensor for non-finite (NaN/Inf) values position-by-position
            the way the `axis=None` path checks the whole tensor once;
            catch NaN/Inf upstream (see `WinCore.diagnostics.record_loss`)
            rather than relying on this function to notice it per-row.

    Returns an `Fp8Tensor(data, scale, orig_dtype, axis)`. Pass it
    whole to `dequantize_fp8()` to recover an approximation of the
    original tensor at `orig_dtype` -- the same call works whether
    `axis` was given or not.
    """
    import torch

    if tensor.numel() == 0:
        raise ValueError(
            "quantize_fp8: got an empty tensor (0 elements) -- there's "
            "no meaningful value range to scale against. This used to "
            "surface as a confusing low-level error from inside "
            "amax() ('zero-size array to reduction operation maximum "
            "which has no identity') instead of an actionable message; "
            "if an empty tensor can legitimately reach this call in "
            "your pipeline, check tensor.numel() before calling."
        )

    dtype = resolve_dtype(f"fp8_{fmt}" if fmt in ("e4m3", "e5m2") else fmt)

    # BUGFIX (v0.8.2, was #8.3): fp8_max used to be picked from the raw
    # `fmt` argument compared against the literal string "e4m3". That
    # only matched when the caller passed exactly "e4m3" -- any of the
    # other spellings `resolve_dtype()` itself accepts ("fp8_e4m3",
    # "float8_e4m3fn") fell through to the `else` branch and silently
    # got e5m2's ceiling (57344.0) instead of e4m3's (448.0). The scale
    # factor was then off by ~128x, so `quantized` overflowed e4m3's
    # actual range on cast with no error raised -- wrong numbers, no
    # warning. Deriving the max from the *resolved* dtype (what will
    # actually be cast to) instead of the raw string closes every alias
    # at once instead of enumerating them.
    is_e5m2 = hasattr(torch, "float8_e5m2") and dtype == torch.float8_e5m2
    fp8_max = 57344.0 if is_e5m2 else 448.0

    orig_dtype = tensor.dtype
    finite = tensor.detach()

    if axis is None:
        amax = finite.abs().amax()
        if amax == 0 or not torch.isfinite(amax):
            # An all-zero (or all-nonfinite, which the caller should
            # have caught earlier -- see diagnostics.record_loss)
            # tensor has no meaningful range to scale against;
            # scale=1.0 is a safe no-op.
            scale = 1.0
        else:
            # Scale so the tensor's own max maps just under fp8_max,
            # instead of a fixed constant that's wrong for most
            # tensors.
            scale = float(amax.item()) / (fp8_max * 0.9)  # 10% headroom
            scale = scale if scale > 0 else 1.0
        quantized = (finite.float() / scale).to(dtype)
        return Fp8Tensor(data=quantized, scale=scale, orig_dtype=orig_dtype, axis=None)

    # -- per-axis path --
    ndim = len(tensor.shape)
    norm_axis = axis if axis >= 0 else axis + ndim
    if not (0 <= norm_axis < ndim):
        # BUGFIX (found in audit): an out-of-range axis used to be
        # silently ACCEPTED instead of rejected -- `reduce_dims` was
        # built as "every dim except norm_axis", and since an
        # out-of-range norm_axis never equals any real dim index, that
        # comprehension silently included EVERY dim (reducing over the
        # whole tensor, identical to axis=None), while still recording
        # the caller's bogus, out-of-range axis value on the returned
        # Fp8Tensor.axis -- a working-looking result with wrong,
        # misleading metadata, not an error. `axis=5` on a 2D tensor,
        # or `axis=-5`, now raise immediately instead of silently
        # degrading to a global scale under a false per-axis label.
        raise ValueError(
            f"quantize_fp8: axis={axis} is out of range for a "
            f"{ndim}-dimensional tensor (valid range: "
            f"{-ndim}..{ndim - 1}). Pass axis=None for whole-tensor "
            f"scaling instead of an axis index."
        )
    reduce_dims = tuple(d for d in range(ndim) if d != norm_axis)
    amax = finite.abs().amax(dim=reduce_dims, keepdim=True) if reduce_dims else finite.abs()
    # Tiny epsilon guards a position whose amax is exactly 0 (an
    # all-zero row/channel) from a 0/0 -> NaN division below: with the
    # epsilon, that position's `scale` is a small positive number
    # instead of exactly zero, so `0 / tiny_scale` correctly comes out
    # as 0, not NaN -- and the epsilon is small enough (1e-12 against
    # real weight/activation magnitudes) to not measurably perturb the
    # scale for any position that WASN'T all-zero.
    scale = amax / (fp8_max * 0.9) + 1e-12
    quantized = (finite.float() / scale).to(dtype)
    return Fp8Tensor(data=quantized, scale=scale, orig_dtype=orig_dtype, axis=norm_axis)


def _pack_4bit_codes(codes: List[int]) -> List[int]:
    """Pure Python, torch-independent: pack a list of 4-bit codes
    (each in `[0, 15]`) two-per-byte into a list of `int`s in
    `[0, 255]`. This is the exact bit arithmetic `quantize_fp4()` uses
    on a real tensor's `.tolist()`'d values -- factored out here so
    the packing/unpacking math itself has direct, deterministic unit
    coverage without needing torch or CUDA at all (same pattern this
    module's `_predict_future_fraction`/`_decide_clear` siblings in
    `WinCore.memory` use for their own pure decision logic).

    If `codes` has an odd length, a `0` code is appended before
    packing (never read back out -- `_unpack_4bit_codes` is always
    called with the real `numel`, which truncates the padding away).
    """
    if len(codes) % 2 == 1:
        codes = codes + [0]
    packed = []
    for i in range(0, len(codes), 2):
        high = codes[i] & 0x0F
        low = codes[i + 1] & 0x0F
        packed.append((high << 4) | low)
    return packed


def _unpack_4bit_codes(packed: List[int], numel: int) -> List[int]:
    """Inverse of `_pack_4bit_codes` -- unpack a list of bytes back
    into `numel` 4-bit codes (dropping any padding code the packer
    added for an odd-length original)."""
    codes = []
    for byte in packed:
        codes.append((byte >> 4) & 0x0F)
        codes.append(byte & 0x0F)
    return codes[:numel]


@dataclass
class Fp4Tensor:
    """Result of `quantize_fp4()` — see that function's docstring for
    the (deliberately loud) caveats before using this for anything but
    experimentation."""

    packed: "object"  # torch.Tensor (uint8 if available on this torch build, else int32), half the byte count of an int8 encoding of the same element count
    scale: float
    orig_shape: tuple
    orig_dtype: "object"
    numel: int  # original element count (packing may round up internally for odd counts; this is the count to actually restore)


def quantize_fp4(tensor) -> Fp4Tensor:
    """EXPERIMENTAL, software-only 4-bit linear quantization: 15
    symmetric levels (-7..+7), two 4-bit codes packed per byte, for a
    real ~8x storage reduction vs fp32 and ~2x vs `quantize_fp8()`'s
    output (4 bits vs 8 bits per element — packing overhead is
    negligible: 1 extra padding code at most, only for an odd element
    count).

    Read this before reaching for it over `quantize_fp8()`
    --------------------------------------------------------
    This module's own `resolve_dtype()` already states the position
    plainly: there is no IEEE/CUDA-native 4-bit compute dtype, and this
    function does not pretend otherwise. Concretely:
      - NO hardware tensor-core acceleration exists for this the way
        fp8 has on Hopper/Ada+. This is pure bit-packing on top of
        ordinary integer ops — there is no "fp4 matmul" this produces;
        using the result in any arithmetic requires
        `dequantize_fp4()` first, unpacking back to a wider dtype.
      - Only 15 representable magnitude levels total (vs fp8 e4m3's
        several hundred distinct finite values) means substantially
        MORE quantization error than `quantize_fp8()` for the same
        tensor. Appropriate for aggressive, accepted-precision-loss
        scenarios (compressing rarely-touched optimizer state further,
        shrinking an already-fp8 KV cache) — NOT a drop-in replacement
        for fp8/fp16 on accuracy-sensitive weights or activations.
      - This is a LINEAR symmetric scheme (uniformly spaced levels),
        not NF4/AF4 (non-uniform, information-theoretically better for
        the roughly-Gaussian weight distributions bitsandbytes/AWQ/
        GPTQ target). If you need production-grade 4-bit weight
        quantization for inference, use one of those dedicated
        libraries — this exists for exploratory experimentation (as
        this module's own module docstring already flags 4-bit as:
        "not offered [as a compute dtype]... a quantization scheme,
        e.g. via bitsandbytes"), not to replace them.
      - The packing/unpacking loop below runs in plain Python over
        `tensor.tolist()`, NOT a vectorized torch op — a deliberate
        trade-off consistent with "experimental, not the recommended
        default": correctness and auditability over throughput. For a
        large tensor this is measurably slower than `quantize_fp8()`.
        If you need this fast enough for a hot path, that's a signal
        this isn't the right tool for that path yet.

    Args:
        tensor: a float16/bfloat16/float32/float64 torch.Tensor of any
            shape.

    Returns an `Fp4Tensor`. Pass it whole to `dequantize_fp4()`.
    """
    import torch

    if tensor.numel() == 0:
        raise ValueError(
            "quantize_fp4: got an empty tensor (0 elements) -- there's "
            "no meaningful value range to scale against. This used to "
            "surface as a confusing low-level error from inside "
            "amax() instead of an actionable message; if an empty "
            "tensor can legitimately reach this call in your "
            "pipeline, check tensor.numel() before calling."
        )

    orig_dtype = tensor.dtype
    orig_shape = tuple(tensor.shape)
    finite = tensor.detach().float()
    flat = finite.flatten()
    numel = int(flat.shape[0])

    amax = flat.abs().amax()
    if amax == 0 or not torch.isfinite(amax):
        scale = 1.0
    else:
        # 15 symmetric levels: -7..+7 (0 is one of them, at code 7 --
        # see the +7 shift below), so the largest magnitude maps to
        # level 7, same "headroom" reasoning as quantize_fp8's fp8_max.
        scale = float(amax.item()) / 7.0
        scale = scale if scale > 0 else 1.0

    values = flat.tolist()
    # Round each scaled value to the nearest of the 15 symmetric
    # integer levels, clamp anything past +/-7 (shouldn't normally
    # happen given `scale` was set from this same tensor's own amax,
    # but a defensive clamp costs nothing and protects against a
    # caller passing a mismatched external scale by hand in future
    # code that reuses these helpers).
    codes = []
    for v in values:
        level = round(v / scale)
        level = max(-7, min(7, level))
        codes.append(level + 7)  # shift to an unsigned 4-bit code, 0..14

    packed_bytes = _pack_4bit_codes(codes)

    uint8_dtype = getattr(torch, "uint8", None)
    packed_tensor = torch.tensor(packed_bytes, dtype=uint8_dtype) if uint8_dtype is not None else torch.tensor(packed_bytes)

    return Fp4Tensor(packed=packed_tensor, scale=scale, orig_shape=orig_shape, orig_dtype=orig_dtype, numel=numel)


def dequantize_fp4(packed: Fp4Tensor):
    """Inverse of `quantize_fp4()` — unpacks the 4-bit codes back to
    `orig_dtype` at `orig_shape`. Lossy in two ways stacked together:
    the quantization error inherent to only 15 levels (see
    `quantize_fp4()`'s docstring), plus the same scale-based
    reconstruction limits `dequantize_fp8()` has. Same Python-loop
    trade-off as `quantize_fp4()` — not vectorized, by design, for the
    same "experimental, not the recommended default" reasons."""
    import torch

    packed_bytes = [int(b) for b in packed.packed.tolist()]
    codes = _unpack_4bit_codes(packed_bytes, packed.numel)
    values = [(code - 7) * packed.scale for code in codes]
    return torch.tensor(values).reshape(packed.orig_shape).to(packed.orig_dtype)


def dequantize_fp8(packed: Fp8Tensor):
    """Inverse of `quantize_fp8()` — restores an approximation of the
    original tensor at its original dtype. Works identically whether
    `packed.scale` is a plain float (whole-tensor quantization) or a
    broadcastable tensor (per-axis quantization, `axis=` was given) —
    the multiplication below broadcasts correctly either way, so this
    function doesn't need to know or care which mode produced
    `packed`. Lossy: fp8's mantissa is too narrow to round-trip
    exactly, by design (that's the whole compression trade-off) --
    this returns the best reconstruction available from what was kept
    (magnitude via `scale`, low-precision values via the fp8 data),
    not the original bits."""
    import torch

    return (packed.data.float() * packed.scale).to(packed.orig_dtype)


def safe_cast(tensor, dtype, on_overflow: str = "warn"):
    """Cast `tensor` to `dtype`, but check for real, silent overflow
    FIRST -- specifically the failure mode `float16` has that
    `bfloat16`/`float32`/`float64` don't: a value whose magnitude
    exceeds float16's representable range (~65504) does not raise or
    clip on cast -- it silently becomes `inf`, with no warning, no
    exception, nothing in the loss/gradient values themselves that
    looks obviously wrong until much later (a NaN a few steps
    downstream from an inf multiplying into something). `bfloat16`
    doesn't have this specific failure mode (its exponent range
    matches float32's -- see `WinCore.precision`'s own module
    docstring on why bf16 is preferred on Ampere+ for exactly this
    reason), and casting DOWN in precision generally (fp32->fp16 for
    values already in-range) is a normal, intended operation this
    function does not flag.

    This directly reduces one of float16's well-known "limitations"
    (see this module's own `recommended_dtype()` docstring, which
    already steers away from float16 where bfloat16 is available) by
    turning a silent, hard-to-trace `inf` into an explicit, immediate
    signal at the exact cast that caused it.

    Args:
        tensor: any torch.Tensor.
        dtype: the target dtype (e.g. `torch.float16`).
        on_overflow: what to do if casting would overflow:
          - `"warn"` (default): cast anyway (matching plain
            `tensor.to(dtype)` behavior exactly), but emit a
            `RuntimeWarning` via the standard `warnings` module first,
            naming the actual max magnitude found and the dtype's
            limit -- visible, but non-fatal, matching this package's
            general default posture of warning rather than crashing a
            run over a recoverable issue (see `WinCore.cpu.apply`'s
            `strict=` parameter for the same philosophy elsewhere).
          - `"raise"`: raise `OverflowError` instead of casting, for
            callers who want this to be a hard stop (e.g. a
            correctness-critical path, or a test asserting no
            silent-overflow ever happens).
          - `"clip"`: clamp the tensor to the target dtype's finite
            range BEFORE casting, so the result has no `inf` at all —
            trades a hard ceiling (values are floored/capped, not
            preserved) for guaranteed-finite output. Prefer this over
            `"warn"` when a stray `inf` would be worse than a clipped
            value (e.g. feeding directly into a loss that would
            otherwise propagate the `inf` before anyone reads the
            warning).

    Only actually checks the dtypes with a KNOWN silent-overflow risk
    in torch (`float16`, `float8_e4m3fn`, `float8_e5m2` — all narrower-
    range than fp32) -- casting to `bfloat16`/`float32`/`float64`
    (matching or wider exponent range than the common source dtypes)
    skips the check entirely and behaves exactly like `tensor.to(dtype)`,
    since there's no overflow risk this function exists to catch there.

    Returns the cast tensor (same as `tensor.to(dtype)` would, in the
    no-overflow and `"warn"` cases).
    """
    import torch
    import warnings

    _OVERFLOW_LIMITS = {}
    if hasattr(torch, "float16"):
        _OVERFLOW_LIMITS[torch.float16] = 65504.0
    if hasattr(torch, "float8_e4m3fn"):
        _OVERFLOW_LIMITS[torch.float8_e4m3fn] = 448.0
    if hasattr(torch, "float8_e5m2"):
        _OVERFLOW_LIMITS[torch.float8_e5m2] = 57344.0

    limit = _OVERFLOW_LIMITS.get(dtype)
    if limit is None:
        # Not one of the dtypes this function knows has a silent
        # overflow-to-inf risk -- behave exactly like a plain cast.
        return tensor.to(dtype)

    amax = tensor.detach().abs().amax()
    amax_value = float(amax.item()) if hasattr(amax, "item") else float(amax)

    if amax_value <= limit:
        return tensor.to(dtype)

    if on_overflow == "raise":
        raise OverflowError(
            f"safe_cast: tensor max magnitude {amax_value:.6g} exceeds "
            f"{dtype_name(dtype)}'s representable range ({limit:.6g}) -- "
            f"casting would silently produce inf. Pass on_overflow="
            f"'clip' to clamp instead, or 'warn' to cast anyway with a "
            f"warning."
        )

    if on_overflow == "clip":
        clipped = tensor.clamp(min=-limit, max=limit) if hasattr(tensor, "clamp") else tensor
        return clipped.to(dtype)

    warnings.warn(
        f"safe_cast: tensor max magnitude {amax_value:.6g} exceeds "
        f"{dtype_name(dtype)}'s representable range ({limit:.6g}) -- "
        f"this cast will silently produce inf for the values above that "
        f"range (torch does not raise or clip on this by default). Pass "
        f"on_overflow='clip' to clamp instead, or 'raise' to make this a "
        f"hard error.",
        RuntimeWarning,
        stacklevel=2,
    )
    return tensor.to(dtype)


@dataclass
class AmpPlan:
    """What `amp()` decided for this GPU. `enabled=False` means autocast
    is a no-op context manager and `scaler.is_enabled()` is False — both
    are still safe to call unconditionally in your training loop, so you
    don't need an `if amp_plan.enabled:` branch in the step function
    itself."""

    enabled: bool
    dtype: "object"
    use_grad_scaler: bool  # True only for float16 (bfloat16 has enough dynamic range that loss scaling isn't needed)
    reason: str


def amp(device_index: int = 0):
    """One call that sets up autocast + `GradScaler` consistently,
    instead of the caller having to separately: pick a dtype, remember
    that `GradScaler` is only meaningful for float16 (not bfloat16 —
    bf16's exponent range doesn't need loss scaling, and enabling the
    scaler for it just adds unnecessary overhead), and remember to
    disable both cleanly on CPU-only runs.

    Returns an `AmpContext` with:
      - `.autocast()` — a `torch.autocast` context manager configured
        with the right device type and dtype for this machine
        (`enabled=False` degrades to a real no-op context manager on
        CPU-only or pre-fp16-tensor-core GPUs, not a crash).
      - `.scaler` — a `torch.amp.GradScaler("cuda", ...)` (falling back
        to the older `torch.cuda.amp.GradScaler` on pre-2.x torch),
        enabled only when
        `dtype is torch.float16` (disabled — a harmless pass-through —
        for bfloat16/float32).
      - `.plan` — the `AmpPlan` explaining what was chosen and why.

    Typical loop:

        ctx = WinCore.precision.amp()
        for batch in loader:
            optimizer.zero_grad()
            with ctx.autocast():
                loss = model(batch)
            ctx.scaler.scale(loss).backward()
            ctx.scaler.step(optimizer)
            ctx.scaler.update()

    This wires the dtype decision from `recommended_dtype()` into
    autocast/GradScaler; it does not change what autocast/GradScaler
    actually do internally, and it does not implement mixed precision
    itself — `torch.autocast` and `torch.amp.GradScaler` do that,
    exactly as they would if you constructed them by hand with the
    same dtype.
    """
    import torch

    dtype = recommended_dtype(device_index)
    cuda_available = torch.cuda.is_available()
    enabled = cuda_available and dtype in (torch.float16, torch.bfloat16)
    use_scaler = enabled and dtype is torch.float16

    if not cuda_available:
        reason = "No CUDA device available — autocast and GradScaler are both no-ops; runs in plain float32."
    elif not enabled:
        reason = f"GPU dtype resolved to {dtype_name(dtype)} — no autocast benefit over plain float32 on this GPU."
    elif use_scaler:
        reason = "float16 autocast with GradScaler enabled (loss scaling needed to avoid float16 underflow in backward)."
    else:
        reason = "bfloat16 autocast enabled; GradScaler left disabled (bfloat16's exponent range doesn't need loss scaling)."

    plan = AmpPlan(enabled=enabled, dtype=dtype, use_grad_scaler=use_scaler, reason=reason)
    return AmpContext(plan)


class AmpContext:
    """Returned by `amp()`. Not meant to be constructed directly."""

    def __init__(self, plan: "AmpPlan"):
        self.plan = plan
        import torch

        # torch.cuda.amp.GradScaler is deprecated in favor of
        # torch.amp.GradScaler("cuda", ...) as of newer torch versions;
        # try the new form first and fall back for older torch that
        # doesn't have it yet, rather than picking one and breaking the
        # other.
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=plan.use_grad_scaler)
        except (AttributeError, TypeError):
            self.scaler = torch.cuda.amp.GradScaler(enabled=plan.use_grad_scaler)

    def autocast(self):
        import torch

        device_type = "cuda" if self.plan.enabled else "cpu"
        return torch.autocast(device_type=device_type, dtype=self.plan.dtype, enabled=self.plan.enabled)


@dataclass
class CudaPerfPlan:
    """What `cuda_perf_defaults()` decided and (if `apply=True`)
    actually set on `torch.backends.*` for this process. Every field
    reflects what was genuinely done, not what was requested -- e.g.
    `tf32` is `False` on a pre-Ampere GPU even if you didn't pass
    `tf32=False`, because setting that flag there is a documented
    no-op (the hardware path it controls doesn't exist), and reporting
    `True` anyway would be a lie about what's actually running.
    """

    applied: bool
    tf32: bool  # matmul + cudnn TF32 -- only meaningful on compute capability >= 8.0
    cudnn_benchmark: bool
    sdpa_backends_enabled: List[str]  # which fused attention backends were turned on, if any
    warnings: List[str]
    reason: str


def cuda_perf_defaults(
    device_index: int = 0,
    tf32: bool = True,
    cudnn_benchmark: bool = True,
    sdpa_backends: bool = True,
    apply: bool = True,
) -> CudaPerfPlan:
    """Turn on the standard, well-documented PyTorch performance knobs
    for this GPU in one call, instead of every training script having
    to remember and re-copy the same handful of `torch.backends.*`
    lines (and re-derive, from scratch, which of them actually apply
    to the GPU that happens to be installed).

    This does not invent any new optimization technique -- every knob
    here is a real, official `torch.backends` switch, documented by
    PyTorch itself. What this function adds is: deciding WHICH of them
    are safe/meaningful to flip for the GPU actually in this machine
    (instead of setting all of them unconditionally and hoping), and
    surfacing that decision as a `CudaPerfPlan` you can log or assert
    on, the same pattern `amp()` above already uses for dtype choice.

    What each flag does, and why the default is what it is
    -----------------------------------------------------------
    - `tf32` (default `True`): TensorFloat-32 lets `torch.matmul`/
      `nn.Linear`/conv ops on the GPU's tensor cores run at
      near-fp16 throughput while keeping fp32 storage and (per
      PyTorch's own docs) fp16-comparable-or-better numerical accuracy
      for typical deep learning workloads. This is PyTorch's own
      recommended default posture for training on Ampere+ (compute
      capability >= 8.0; RTX 30-series and newer, A100, H100, etc.) --
      *NOT* something this package invents; PyTorch just doesn't
      enable it by default itself, historically for backward-numerics
      reasons. Setting `torch.backends.cuda.matmul.allow_tf32` /
      `torch.backends.cudnn.allow_tf32` on older hardware is
      documented as a harmless no-op (the instruction path doesn't
      exist there) -- but this function still checks compute
      capability first and reports `tf32=False` in that case, rather
      than claiming a flag did something it structurally couldn't.
      Also sets `torch.set_float32_matmul_precision("high")` where
      available (torch>=1.12) -- the newer, preferred spelling of the
      same intent that some ops key off instead of the raw
      `allow_tf32` booleans.
    - `cudnn_benchmark` (default `True`): tells cuDNN to benchmark
      several convolution algorithms the first time it sees each
      distinct input shape, then cache and reuse the fastest one for
      that shape thereafter. Real, meaningful speedup for the common
      case of fixed input shapes (e.g. constant batch size and image
      resolution) across a training run. The trade-off, stated plainly
      rather than hidden: (1) the first iteration at each distinct
      shape pays autotuning overhead instead of running training
      immediately, and (2) if your input shapes actually DO vary every
      step (variable-length sequences without padding to a fixed
      bucket, e.g.), cuDNN re-benchmarks on every new shape it sees,
      which can make training *slower* than leaving this off, not
      faster. Pass `cudnn_benchmark=False` if that's your workload --
      this function does not try to detect variable-shape usage
      itself, since that's a property of your data pipeline this
      module has no visibility into.
    - `sdpa_backends` (default `True`): explicitly enables all three
      of PyTorch's built-in `scaled_dot_product_attention` backends
      (Flash Attention, memory-efficient attention, and the plain math
      fallback) via `torch.backends.cuda.enable_*_sdp()`, so
      `F.scaled_dot_product_attention` (used directly, or indirectly
      by anything built on it) can pick the fastest one actually
      available for the given shapes/dtype/GPU at call time instead of
      being artificially restricted to whichever backend happened to
      be enabled by torch's own default for that version. This is
      inert (no-op, not an error) on a torch build old enough not to
      have these toggles at all, and on non-attention workloads that
      never call `scaled_dot_product_attention`.

    Args:
        device_index: which GPU's compute capability to check for the
            `tf32` decision (only matters on a multi-GPU machine with
            mixed GPU generations -- rare, but this function doesn't
            assume otherwise).
        tf32, cudnn_benchmark, sdpa_backends: set any to `False` to
            skip that specific knob (e.g. `cudnn_benchmark=False` for
            a variable-input-shape workload, as above), while still
            getting the others.
        apply: if `True` (default), actually sets the `torch.backends`
            flags. If `False`, returns the `CudaPerfPlan` that WOULD be
            applied without touching anything -- useful for logging
            the plan before committing to it, or in a dry-run/`--info`
            CLI path.

    Returns a `CudaPerfPlan` describing exactly what was (or would be)
    changed and why. Safe to call with no CUDA device present: returns
    an all-`False`/empty plan with `reason` explaining that, rather
    than raising.
    """
    import torch

    warnings: List[str] = []

    if not torch.cuda.is_available():
        return CudaPerfPlan(
            applied=False,
            tf32=False,
            cudnn_benchmark=False,
            sdpa_backends_enabled=[],
            warnings=warnings,
            reason="No CUDA device available -- nothing to tune.",
        )

    major, _minor = torch.cuda.get_device_capability(device_index)
    tf32_capable = major >= 8
    do_tf32 = tf32 and tf32_capable
    if tf32 and not tf32_capable:
        warnings.append(
            f"tf32 requested but compute capability {major}.x < 8.0 "
            f"(pre-Ampere) -- TF32 tensor-core paths don't exist on this "
            f"GPU, so this flag would be a documented no-op; left unset."
        )

    enabled_sdp: List[str] = []
    if sdpa_backends:
        for attr_name, label in (
            ("enable_flash_sdp", "flash"),
            ("enable_mem_efficient_sdp", "mem_efficient"),
            ("enable_math_sdp", "math"),
        ):
            fn = getattr(torch.backends.cuda, attr_name, None)
            if fn is None:
                warnings.append(
                    f"torch.backends.cuda.{attr_name} not present on this "
                    f"torch build -- skipped ({label} SDPA backend toggle "
                    f"unavailable, not an error)."
                )
                continue
            if apply:
                fn(True)
            enabled_sdp.append(label)

    if apply:
        torch.backends.cuda.matmul.allow_tf32 = do_tf32
        torch.backends.cudnn.allow_tf32 = do_tf32
        set_precision = getattr(torch, "set_float32_matmul_precision", None)
        if set_precision is not None:
            set_precision("high" if do_tf32 else "highest")
        torch.backends.cudnn.benchmark = cudnn_benchmark

    reason_parts = []
    reason_parts.append(
        f"TF32 {'enabled' if do_tf32 else 'left off'} (compute capability {major}.x)."
    )
    reason_parts.append(
        f"cuDNN benchmark {'enabled' if cudnn_benchmark else 'left off'}."
    )
    if sdpa_backends:
        reason_parts.append(
            f"SDPA backends enabled: {enabled_sdp or 'none available on this torch build'}."
        )
    reason = " ".join(reason_parts)

    return CudaPerfPlan(
        applied=apply,
        tf32=do_tf32,
        cudnn_benchmark=cudnn_benchmark,
        sdpa_backends_enabled=enabled_sdp,
        warnings=warnings,
        reason=reason,
    )
