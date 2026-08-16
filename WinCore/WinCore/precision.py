"""
Precision helper — pick a safe torch dtype instead of guessing.

Why this exists
----------------
`bfloat16` is only fast (or supported at all) on GPUs with compute
capability >= 8.0 (NVIDIA Ampere / RTX 30-series and newer). Using it on
older cards silently falls back to slow emulation or errors depending on
the op. `float16` works further back but can overflow in some training
loops. This module answers one narrow question honestly: "given the GPU
that's actually in this machine, which dtype should I default to?" —
it does not claim to support fp4 as a native compute dtype (no such
IEEE/CUDA-native type exists broadly; 4-bit is a *quantization* scheme,
e.g. via bitsandbytes, layered on top of int8/fp16 storage — that's a
different, separate technique from choosing a compute dtype here).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


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
    per-tensor scale needed to bring it back to the original range.
    fp8 (e4m3/e5m2) has only ~2-3 bits of mantissa, so unlike fp16/bf16
    it cannot store an arbitrary-magnitude tensor directly without
    clipping or flushing small values to zero -- a *scale* factor is
    required to fit the tensor's actual value range into fp8's narrow
    representable range before casting down, and to undo that scaling
    on the way back up. Storing `data` and `scale` together is what
    makes this a lossy-but-controlled compression scheme rather than
    "just cast to a smaller dtype and hope"."""

    data: "object"  # torch.Tensor, dtype=float8_e4m3fn or float8_e5m2
    scale: float
    orig_dtype: "object"  # the dtype to restore on dequantize


def quantize_fp8(tensor, fmt: str = "e4m3") -> Fp8Tensor:
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

    Returns an `Fp8Tensor(data, scale, orig_dtype)`. Pass it whole to
    `dequantize_fp8()` to recover an approximation of the original
    tensor at `orig_dtype`.
    """
    import torch

    dtype = resolve_dtype(f"fp8_{fmt}" if fmt in ("e4m3", "e5m2") else fmt)

    orig_dtype = tensor.dtype
    finite = tensor.detach()
    amax = finite.abs().amax()
    if amax == 0 or not torch.isfinite(amax):
        # An all-zero (or all-nonfinite, which the caller should have
        # caught earlier -- see diagnostics.record_loss) tensor has no
        # meaningful range to scale against; scale=1.0 is a safe no-op.
        scale = 1.0
    else:
        # fp8 e4m3's largest finite magnitude is 448; e5m2's is 57344.
        # Scale so the tensor's own max maps just under that ceiling,
        # instead of a fixed constant that's wrong for most tensors.
        fp8_max = 448.0 if fmt == "e4m3" else 57344.0
        scale = float(amax.item()) / (fp8_max * 0.9)  # 10% headroom
        scale = scale if scale > 0 else 1.0

    quantized = (finite.float() / scale).to(dtype)
    return Fp8Tensor(data=quantized, scale=scale, orig_dtype=orig_dtype)


def dequantize_fp8(packed: Fp8Tensor):
    """Inverse of `quantize_fp8()` — restores an approximation of the
    original tensor at its original dtype. Lossy: fp8's mantissa is
    too narrow to round-trip exactly, by design (that's the whole
    compression trade-off) -- this returns the best reconstruction
    available from what was kept (magnitude via `scale`, low-precision
    values via the fp8 data), not the original bits."""
    import torch

    return (packed.data.float() * packed.scale).to(packed.orig_dtype)


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
