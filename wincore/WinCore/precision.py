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
