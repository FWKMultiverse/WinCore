"""
Gradient accumulation — correct loss scaling + DDP `no_sync()`
skipping on non-boundary micro-steps.

Why this exists
----------------
Gradient accumulation (running several forward/backward passes on
smaller micro-batches before one `optimizer.step()`) is the standard
way to train at an effectively larger batch size than fits in VRAM at
once. Two real, easy-to-get-silently-wrong details determine whether
it's correct AND fast:

  1. LOSS SCALING: each micro-batch's loss must be divided by the
     number of accumulation steps before `.backward()`, or the summed
     gradient magnitude across N micro-steps ends up N times too
     large — equivalent to training at N times the intended learning
     rate. This is a correctness bug, not a performance one, and it's
     silent: no error, no crash, training "runs" — just with an
     effectively wrong LR for the duration.
  2. DDP COMMUNICATION: with `DistributedDataParallel`, every
     `.backward()` call triggers a full gradient all-reduce across
     every GPU in the run by default — correct, but wasteful when
     you're about to call `.backward()` again on the next micro-batch
     before ever stepping the optimizer. `model.no_sync()` (a real,
     official DDP context manager) skips that all-reduce for every
     micro-step except the last one in the accumulation window, so N
     micro-steps pay for exactly ONE all-reduce instead of N — a real,
     substantial reduction in cross-GPU communication for multi-GPU
     training, not a micro-optimization.

What this module does NOT do
-------------------------------
Does not call `.backward()`, `optimizer.step()`, or
`optimizer.zero_grad()` for you — you still write your own training
loop; this only tells you (a) what to divide the loss by, and (b)
which sync context to run the backward pass under, and (c) whether
this micro-step completes the window (time to actually step). It does
not wrap or replace your model, optimizer, or DDP setup, and it does
not itself import torch — `no_sync()` is called on whatever `model`
object you pass in, duck-typed, so this module works without torch
installed at all until you actually give it a real DDP-wrapped model.
"""
from __future__ import annotations

import contextlib
from typing import Any, Optional


class GradientAccumulator:
    """Tracks micro-step position within an accumulation window and
    supplies the two things needed to make gradient accumulation both
    numerically correct and (under DDP) communication-efficient.

    Example (single GPU / no DDP):
        accum = WinCore.accumulate.GradientAccumulator(accumulation_steps=4)
        for micro_batch in micro_batches:
            loss = model(micro_batch)
            accum.scale_loss(loss).backward()
            if accum.step():          # True only on the 4th micro-step
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

    Example (DDP, with no_sync() skipping on non-boundary steps):
        accum = WinCore.accumulate.GradientAccumulator(accumulation_steps=4, model=ddp_model)
        for micro_batch in micro_batches:
            with accum.sync_context():
                loss = model(micro_batch)
                accum.scale_loss(loss).backward()
            if accum.step():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
    """

    def __init__(self, accumulation_steps: int = 1, model: Optional[Any] = None):
        # BUGFIX (found in audit): a non-integer accumulation_steps
        # (e.g. from `total_batch_size // micro_batch_size` done with
        # true division instead of floor division, or any other
        # arithmetic that doesn't land on a whole number) used to be
        # silently ACCEPTED. `step()`'s boundary check
        # (`self._micro_step >= self.accumulation_steps`) then produces
        # an inconsistent window size that alternates instead of a
        # fixed N -- e.g. accumulation_steps=2.5 gives windows of 3,
        # 2, 3, 2, ... micro-steps, not a stable 2 or 3. That directly
        # undermines the one correctness guarantee this class exists
        # to provide (see the module docstring's "LOSS SCALING" point)
        # -- `scale_loss()` would keep dividing by 2.5 every time while
        # the actual window size silently varies step to step. Reject
        # this at construction instead of producing a subtly-wrong
        # effective batch size that only shows up as unexplained
        # training variance much later.
        if not isinstance(accumulation_steps, int) or isinstance(accumulation_steps, bool):
            raise TypeError(
                f"accumulation_steps must be an int, got "
                f"{type(accumulation_steps).__name__} ({accumulation_steps!r}) "
                f"-- a non-whole-number accumulation window produces an "
                f"inconsistent number of micro-steps per optimizer.step() "
                f"instead of a fixed size, silently breaking the loss-scaling "
                f"guarantee this class exists for."
            )
        if accumulation_steps < 1:
            raise ValueError(f"accumulation_steps must be >= 1, got {accumulation_steps}")
        self.accumulation_steps = accumulation_steps
        self.model = model
        self._micro_step = 0  # 0-indexed position within the CURRENT window

    def scale_loss(self, loss):
        """Divide `loss` by `accumulation_steps` — call `.backward()`
        on the RESULT of this, not on the raw loss. See this module's
        docstring for why skipping this silently changes the
        effective learning rate rather than raising an error (there is
        nothing about a mis-scaled loss that looks like an error from
        inside a single step)."""
        return loss / self.accumulation_steps

    def is_boundary(self) -> bool:
        """`True` if the NEXT `step()` call will complete the current
        accumulation window (i.e. the upcoming micro-step is the one
        that should actually run a synced backward pass). Read-only —
        does not advance the internal counter. `sync_context()` needs
        this answer BEFORE `.backward()` runs, which is why it's a
        separate query from `step()` (which only reports the boundary
        AFTER advancing, once it's too late to have chosen the sync
        context for that same micro-step)."""
        return self._micro_step == self.accumulation_steps - 1

    def sync_context(self):
        """Context manager to wrap ONE micro-step's forward+backward
        pass in.

        With DDP (`model` was given at construction and exposes a
        `no_sync()` method — duck-typed, not an `isinstance` check, so
        any object with that method works, not only
        `torch.nn.parallel.DistributedDataParallel` specifically):
        returns `model.no_sync()` on every micro-step EXCEPT the
        boundary one (see `is_boundary()`), and a real synced pass
        (`contextlib.nullcontext()` — the all-reduce actually runs) on
        the boundary step, so gradients accumulated across the whole
        window are correctly all-reduced exactly once, right before
        `optimizer.step()`.

        Without DDP (`model` is `None`, or doesn't have a `no_sync()`
        method — e.g. a plain `nn.Module` on a single GPU): always
        returns `contextlib.nullcontext()`, a harmless no-op, since
        there's no cross-process gradient sync to skip in the first
        place — safe to call unconditionally regardless of whether
        you're actually running distributed.
        """
        if self.model is not None and hasattr(self.model, "no_sync") and not self.is_boundary():
            return self.model.no_sync()
        return contextlib.nullcontext()

    def step(self) -> bool:
        """Advance the micro-step counter by one. Returns `True`
        exactly when this call completes an accumulation window (this
        was the boundary micro-step) — the signal to call
        `optimizer.step()` / `optimizer.zero_grad()` now. Returns
        `False` for every other micro-step ("keep accumulating, don't
        step the optimizer yet"). Automatically wraps back to the
        start of a new window after returning `True` — no separate
        reset needed between consecutive windows in a normal loop."""
        self._micro_step += 1
        if self._micro_step >= self.accumulation_steps:
            self._micro_step = 0
            return True
        return False

    def reset(self) -> None:
        """Reset the micro-step counter to the start of a fresh
        window. Useful after handling an exception mid-window (so the
        next iteration doesn't inherit a partial count), or between
        epochs if you specifically want each epoch to start on a clean
        boundary rather than continuing a partial window carried over
        from the previous epoch's tail."""
        self._micro_step = 0
