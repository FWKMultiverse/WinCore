"""
Training diagnostics — catches silent ("no crash, but something's off")
training problems, and reports bottleneck data, without changing how
you train.

Design principle: everything here is *read-only observation* on data
you already have (loss value, gradients, time spent). It never changes
your optimizer, your model, your data pipeline, or your hyperparameters
— it only tells you what it noticed, with `on_issue` as a callback so
you decide what to do about it. That's the "help around the edges
without touching what the developer wants" boundary from your last
message, made concrete: this module never calls `.step()`, never
modifies `.grad`, never touches your DataLoader.

What it actually detects (all real, all testable):
  - NaN / Inf in the loss the moment it appears (`record_loss`) — the
    classic "training silently produced garbage 200 steps ago and you
    only noticed at eval time" problem.
  - Loss plateaus: no meaningful improvement over a rolling window
    (`record_loss`) — doesn't diagnose *why*, just flags "stuck".
  - Exploding / vanishing gradients (`record_grad_norm`) — tracks the
    gradient-norm history and flags when it jumps by a large factor
    step-to-step (explode) or stays near-zero for a long stretch
    (vanish/dead).
  - Dataloader vs. compute bottleneck (`data_timer` / `compute_timer`)
    — measures where wall-clock time is actually going. A model can
    train "fine" (no errors, loss goes down) while still wasting most
    of its wall-clock time waiting on the DataLoader — this is
    invisible unless you time it directly, which is what these two
    context managers do.
  - GPU launch/sync stalls (`gpu_timer`) — measures actual GPU-clock
    busy time (via `torch.cuda.Event`) against CPU wall-clock time
    spent in the same block. A tool like `nvidia-smi` reporting "90%
    utilization" is sampled coarsely and doesn't reveal a GPU that's
    idling in short gaps between many small kernel launches, or
    stalling on a host-device sync point (`.item()`, `.cpu()`,
    data-dependent control flow) — `gpu_timer()` catches that gap
    directly, from inside the training loop, without a full profiler
    attached.
  - Cross-signal co-occurrence (`record_signal`) — feed in readings
    from other WinCore modules (GPU temp from `thermal`, VRAM pressure
    from `spec`/`memory`) or your own code, and any loss/gradient Issue
    emitted near that step gets annotated with what else was going on
    at the time (e.g. "grad norm exploded, and a VRAM-pressure spike
    was recorded 1 step earlier"). This is a same-window co-occurrence
    note, not a causal claim — it surfaces signals that were already
    being recorded separately, side by side, instead of leaving you to
    cross-reference timestamps across different logs by hand.
  - Training phase / progress tracking (`record_step_time`) — answers
    "has training actually started making steady progress yet, or is
    it still in the noisy startup window (cuDNN benchmark autotuning,
    kernel JIT/compile, allocator cache warmup), and given how fast
    steps are actually landing right now, how much wall-clock time is
    left". Classifies each step as `warmup` / `steady_state` /
    `stalled` and, if you pass a target step count, reports an ETA
    computed only from steady-state timing (warmup and stall outliers
    excluded, so one slow first step or one stall doesn't skew the
    estimate).

What it does NOT do:
  - Does not explain *why* a problem happened (e.g. it won't tell you
    "your learning rate is too high") — it flags the symptom
    (exploding grad norm) with enough numbers for you to diagnose the
    cause yourself, because guessing the cause without seeing your
    actual model/data would be fabricating a diagnosis.
  - `record_grad_norm` needs torch (imported lazily). NaN/Inf checks on
    a plain float loss, plateau detection, and the timers do not need
    torch at all and work with plain Python numbers.
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, List, Optional, Any


@dataclass
class Issue:
    step: Optional[int]
    code: str
    severity: str  # "warning" | "critical"
    message: str
    data: dict = field(default_factory=dict)


class _StageTimer:
    """Context manager that accumulates elapsed time into a counter on
    the owning TrainingMonitor. Not meant to be constructed directly —
    use `monitor.data_timer()` / `monitor.compute_timer()`."""

    def __init__(self, monitor: "TrainingMonitor", stage: str):
        self._monitor = monitor
        self._stage = stage
        self._t0 = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = time.perf_counter() - self._t0
        if self._stage == "data":
            self._monitor._data_seconds += elapsed
        else:
            self._monitor._compute_seconds += elapsed
        return False


class _GpuStageTimer:
    """Context manager around a compute block that measures *actual GPU
    execution time* (via `torch.cuda.Event`, which times the GPU's own
    clock between two points it executes, not the CPU wall-clock) next
    to the CPU wall-clock time already spent by `compute_timer()`. Not
    meant to be constructed directly — use `monitor.gpu_timer()`.

    Why this is a different measurement than `compute_timer()`: CUDA
    kernel launches are asynchronous, so a CPU thread inside
    `compute_timer()` can be "in the compute block" while the GPU is
    actually idle, waiting for work to be enqueued (Python/launch
    overhead), or blocked on a host-device synchronization point (e.g.
    `.item()`, `.cpu()`, a size-dependent control-flow branch). A high
    "GPU%" reading from `nvidia-smi` does not distinguish real compute
    from a GPU that is repeatedly stalling and un-stalling on a fast
    cycle — CUDA events measure wall time on the GPU's own timeline
    between the recorded points, which is the only way to see that gap
    from inside Python without a profiler attached."""

    def __init__(self, monitor: "TrainingMonitor"):
        self._monitor = monitor
        self._t0 = 0.0
        self._start_evt = None
        self._end_evt = None

    def __enter__(self):
        self._t0 = time.perf_counter()
        try:
            import torch

            if torch.cuda.is_available():
                self._start_evt = torch.cuda.Event(enable_timing=True)
                self._end_evt = torch.cuda.Event(enable_timing=True)
                self._start_evt.record()
        except ImportError:
            # No torch installed at all -- degrade to a wall-clock-only
            # no-op, same as the rest of this module staying importable
            # and usable without torch.
            pass
        return self

    def __exit__(self, exc_type, exc, tb):
        wall = time.perf_counter() - self._t0
        self._monitor._gpu_block_wall_seconds += wall
        if self._start_evt is not None:
            self._end_evt.record()
            # Only synchronize here (once per recorded block, not per
            # kernel) so this stays usable inside a real training loop
            # instead of forcing a sync on every op.
            self._end_evt.synchronize()
            gpu_ms = self._start_evt.elapsed_time(self._end_evt)
            self._monitor._gpu_busy_seconds += gpu_ms / 1000.0
            self._monitor._gpu_timer_samples += 1
        return False


def _default_warmup_steps(expected_param_count: Optional[int]) -> int:
    """Heuristic starting point for how many initial steps to exclude
    from steady-state step timing while cuDNN benchmark autotuning,
    kernel JIT/compile (if using `torch.compile`), and CUDA allocator
    cache warmup settle down -- all of which make the first several
    steps of a run genuinely, legitimately slower than steady state,
    not a problem to fix.

    This is an UNTUNED heuristic, not a measured constant -- stated
    plainly rather than dressed up as more precise than it is. It
    scales coarsely with `expected_param_count` because a larger model
    touches more distinct op shapes/algorithms for cuDNN's benchmark
    mode to search through, and a larger `torch.compile` graph takes
    longer to first-compile -- both real, directionally-correct
    reasons a bigger model's startup window tends to run longer, not a
    precisely fitted curve. If you have a better number for your own
    setup (e.g. from watching one run once and seeing where step time
    actually stabilizes), pass `warmup_steps=` explicitly to
    `TrainingMonitor(...)` and this heuristic is not consulted at all.
    """
    if expected_param_count is None:
        return 10
    if expected_param_count < 10_000_000:  # < 10M params
        return 5
    if expected_param_count < 1_000_000_000:  # < 1B params
        return 15
    return 30  # >= 1B params


@dataclass
class PhaseStatus:
    """Returned by `record_step_time()`. A snapshot of where training
    is *right now* in terms of timing/progress, not a historical log
    (see `monitor.issues` for the log of `stalled` events, which are
    also emitted as `Issue`s)."""

    phase: str  # "warmup" | "steady_state" | "stalled"
    step: int
    steps_recorded: int
    elapsed_seconds: float  # wall-clock since the FIRST record_step_time() call
    last_step_seconds: float
    steady_state_avg_seconds: Optional[float]  # None until >=1 non-warmup step recorded
    steps_per_second: Optional[float]  # None until steady_state_avg_seconds is known
    eta_seconds: Optional[float]  # None unless total_steps was given AND steady state is known
    reason: str


class TrainingMonitor:
    """Attach to a training loop to catch silent problems and report
    bottleneck timing. Every `record_*` call is O(1)-ish and safe to
    call every step; it never raises on your behalf and never touches
    your model/optimizer/data — it only calls `on_issue` (if you gave
    one) and keeps a log you can inspect with `.issues`.
    """

    def __init__(
        self,
        loss_plateau_window: int = 50,
        loss_plateau_min_relative_improvement: float = 0.001,
        grad_explode_factor: float = 10.0,
        grad_vanish_threshold: float = 1e-7,
        grad_vanish_patience: int = 20,
        signal_correlation_window: int = 3,
        warmup_steps: Optional[int] = None,
        expected_param_count: Optional[int] = None,
        stall_factor: float = 3.0,
        stall_min_samples: int = 5,
        on_issue: Optional[Callable[[Issue], None]] = None,
    ):
        self.loss_plateau_window = loss_plateau_window
        self.loss_plateau_min_relative_improvement = loss_plateau_min_relative_improvement
        self.grad_explode_factor = grad_explode_factor
        self.grad_vanish_threshold = grad_vanish_threshold
        self.grad_vanish_patience = grad_vanish_patience
        self.signal_correlation_window = signal_correlation_window
        # See _default_warmup_steps() docstring: this is a heuristic
        # starting point, not a measured value. `warmup_steps=` (if
        # given explicitly) always wins over the expected_param_count-
        # based guess.
        self.warmup_steps = (
            warmup_steps if warmup_steps is not None else _default_warmup_steps(expected_param_count)
        )
        self.stall_factor = stall_factor
        self.stall_min_samples = stall_min_samples
        self.on_issue = on_issue

        self._loss_history: Deque[float] = deque(maxlen=loss_plateau_window)
        self._grad_norm_history: Deque[float] = deque(maxlen=50)
        self._vanish_streak = 0
        self._data_seconds = 0.0
        self._compute_seconds = 0.0
        self._gpu_block_wall_seconds = 0.0
        self._gpu_busy_seconds = 0.0
        self._gpu_timer_samples = 0
        self.issues: List[Issue] = []
        self._signal_history: Deque[dict] = deque(maxlen=200)

        # -- step-time / phase tracking (record_step_time) --
        self._step_time_start: Optional[float] = None
        self._last_step_ts: Optional[float] = None
        self._steps_recorded = 0
        self._steady_state_step_seconds: Deque[float] = deque(maxlen=200)

    # -- internal -----------------------------------------------------
    def _emit(self, issue: Issue) -> None:
        nearby = self._nearby_signals(issue.step)
        if nearby:
            issue.data["nearby_signals"] = nearby
            names = ", ".join(f"{s['name']}={s['value']!r}" for s in nearby)
            issue.message += (
                f" Other signals recorded around the same step: {names} -- "
                f"worth checking whether they're related (this is a "
                f"co-occurrence in the recorded window, not a proven cause)."
            )
        self.issues.append(issue)
        if self.on_issue is not None:
            self.on_issue(issue)

    def _nearby_signals(self, step: Optional[int]) -> List[dict]:
        """Signals recorded within `signal_correlation_window` steps of
        `step` (or the most recent ones, if `step` is None -- e.g. the
        bottleneck report, which isn't tied to one step)."""
        if not self._signal_history:
            return []
        if step is None:
            return list(self._signal_history)[-3:]
        window = self.signal_correlation_window
        return [
            s for s in self._signal_history
            if s["step"] is not None and abs(s["step"] - step) <= window
        ]

    # -- external signals ------------------------------------------------
    def record_signal(self, step: Optional[int], name: str, value, note: Optional[str] = None) -> None:
        """Feed in a reading from OUTSIDE this monitor -- GPU temperature
        (`WinCore.thermal`), VRAM/RAM pressure (`WinCore.spec`,
        `WinCore.memory.CacheGuard`), a dataloader stall, an optimizer
        LR change, or anything else you want considered when an Issue is
        emitted. Purely observational and cheap (appends to a bounded
        deque) -- it never raises an Issue by itself.

        When a loss/gradient Issue is later emitted at a nearby step
        (within `signal_correlation_window`, default 3), it's annotated
        with whichever signals were recorded around that same step --
        e.g. a gradient explosion that happened right after a VRAM-
        pressure spike or a thermal-guard pause becomes visible together
        instead of as two unrelated log lines you'd have to cross-
        reference by hand. This is a same-window co-occurrence note, not
        causal inference -- it doesn't claim the signal *caused* the
        issue, only that they happened close together, which is real,
        checkable information either way."""
        self._signal_history.append({"step": step, "name": name, "value": value, "note": note})

    # -- loss -----------------------------------------------------------
    def record_loss(self, step: int, loss_value: float) -> Optional[Issue]:
        """Feed the scalar loss for this step. Detects NaN/Inf
        immediately, and a plateau once `loss_plateau_window` values
        have accumulated."""
        if loss_value is None or math.isnan(loss_value) or math.isinf(loss_value):
            issue = Issue(
                step=step,
                code="loss_nan_or_inf",
                severity="critical",
                message=(
                    f"Loss is {loss_value} at step {step}. Training has silently "
                    f"produced garbage from this point on -- typical causes: "
                    f"learning rate too high, unscaled loss in mixed precision, "
                    f"a division/log of zero somewhere in the loss computation."
                ),
                data={"loss": loss_value},
            )
            self._emit(issue)
            return issue

        self._loss_history.append(loss_value)
        if len(self._loss_history) == self.loss_plateau_window:
            first = self._loss_history[0]
            last = self._loss_history[-1]
            denom = abs(first) if abs(first) > 1e-12 else 1e-12
            relative_change = (first - last) / denom
            # BUGFIX (v0.8.2, was #8.2): `relative_change` is signed --
            # positive when loss fell (improving), negative when loss
            # rose (regressing). The old code used one condition
            # (`relative_change < threshold`) for both "barely moved"
            # (relative_change near 0) AND "got much worse"
            # (relative_change very negative, e.g. loss 0.687 -> 3.569
            # gives roughly -419%), then always printed the plateau
            # message ("Loss barely moved ... likely stuck") even when
            # loss had actually blown up. These are opposite situations
            # needing opposite fixes (LR too low/converged vs. LR too
            # high/diverging), so they now get distinct codes/messages.
            # A regression only counts once it's a real move, not noise
            # around zero -- reuse the plateau threshold's magnitude as
            # that noise floor so a tiny negative wobble still reports
            # as a plateau, not a false "regressing" alarm.
            if relative_change < 0 and abs(relative_change) > abs(self.loss_plateau_min_relative_improvement):
                issue = Issue(
                    step=step,
                    code="loss_regressing",
                    severity="warning",
                    message=(
                        f"Loss got WORSE over the last {self.loss_plateau_window} "
                        f"steps ({first:.6g} -> {last:.6g}, "
                        f"{-relative_change*100:.3f}% relative regression). Not "
                        f"stuck -- actively diverging. Typical causes: learning "
                        f"rate too high, a bad batch/data corruption, loss scale "
                        f"too aggressive in mixed precision, or a schedule change "
                        f"(e.g. LR warmup restart) that landed badly."
                    ),
                    data={"first": first, "last": last, "relative_change": relative_change},
                )
                self._emit(issue)
                return issue
            if relative_change < self.loss_plateau_min_relative_improvement:
                issue = Issue(
                    step=step,
                    code="loss_plateau",
                    severity="warning",
                    message=(
                        f"Loss barely moved over the last {self.loss_plateau_window} "
                        f"steps ({first:.6g} -> {last:.6g}, "
                        f"{relative_change*100:.3f}% relative change). Not a crash, "
                        f"but likely stuck -- check learning rate schedule, whether "
                        f"gradients are actually flowing (see record_grad_norm), or "
                        f"whether the model has converged for real."
                    ),
                    data={"first": first, "last": last, "relative_change": relative_change},
                )
                self._emit(issue)
                return issue
        return None

    # -- gradients --------------------------------------------------------
    def record_grad_norm(self, step: int, model) -> Optional[Issue]:
        """Compute the total gradient norm across `model.parameters()`
        (call AFTER `.backward()`, BEFORE `.step()`/`.zero_grad()`) and
        check it against its own recent history for explosion/vanishing.
        Requires torch (imported lazily)."""
        import torch

        total_sq = 0.0
        any_grad = False
        for p in model.parameters():
            if p.grad is not None:
                any_grad = True
                total_sq += float(torch.sum(p.grad.detach() ** 2))
        if not any_grad:
            return None
        norm = math.sqrt(total_sq)

        issue = None
        if self._grad_norm_history:
            prev = self._grad_norm_history[-1]
            if prev > 1e-12 and norm > prev * self.grad_explode_factor:
                issue = Issue(
                    step=step,
                    code="grad_norm_explosion",
                    severity="critical",
                    message=(
                        f"Gradient norm jumped {norm/prev:.1f}x in one step "
                        f"({prev:.4g} -> {norm:.4g}). Typical causes: learning "
                        f"rate too high, a bad batch (outlier/corrupt data), or "
                        f"missing gradient clipping. Loss may still look fine for "
                        f"a few more steps before visibly diverging."
                    ),
                    data={"prev_norm": prev, "norm": norm},
                )

        if norm < self.grad_vanish_threshold:
            self._vanish_streak += 1
            if self._vanish_streak == self.grad_vanish_patience:
                issue = Issue(
                    step=step,
                    code="grad_norm_vanishing",
                    severity="warning",
                    message=(
                        f"Gradient norm has stayed below {self.grad_vanish_threshold:g} "
                        f"for {self.grad_vanish_patience} consecutive steps (currently "
                        f"{norm:.4g}). Typical causes: saturated activations "
                        f"(sigmoid/tanh deep stacks), dead ReLUs, or a learning rate "
                        f"effectively zero after scheduling."
                    ),
                    data={"norm": norm, "streak": self._vanish_streak},
                )
        else:
            self._vanish_streak = 0

        self._grad_norm_history.append(norm)
        if issue is not None:
            self._emit(issue)
        return issue

    # -- bottleneck timing --------------------------------------------
    def data_timer(self) -> _StageTimer:
        """`with monitor.data_timer(): batch = next(loader_iter)`"""
        return _StageTimer(self, "data")

    def compute_timer(self) -> _StageTimer:
        """`with monitor.compute_timer(): loss = train_step(batch)`"""
        return _StageTimer(self, "compute")

    def gpu_timer(self) -> _GpuStageTimer:
        """`with monitor.gpu_timer(): loss = train_step(batch)` — wrap
        the same block you'd wrap in `compute_timer()` (they can be
        nested/used together) to additionally measure real GPU-clock
        busy time via `torch.cuda.Event`, not just CPU wall-clock time
        spent inside the block. See `_GpuStageTimer` for why these two
        numbers can diverge, and why that divergence is exactly the
        "GPU utilization% looks fine but real throughput is low" gap.

        Costs one `event.synchronize()` per call (a real, small stall
        — it waits for the GPU to finish the block before returning),
        so this is meant for periodic diagnostic sampling (e.g. every
        N steps), not necessarily every single step of a long run,
        though calling it every step is also correct, just not free."""
        return _GpuStageTimer(self)

    def bottleneck_report(self) -> dict:
        """Return accumulated data-wait vs. compute time and, if data
        wait is >=40% of total, an Issue flagging the DataLoader as the
        likely bottleneck (a real, common, otherwise-invisible problem
        -- training can look completely fine, no errors, loss going
        down, while most wall-clock time is spent waiting on the
        DataLoader instead of on the GPU)."""
        total = self._data_seconds + self._compute_seconds
        data_fraction = (self._data_seconds / total) if total > 0 else 0.0
        report = {
            "data_seconds": self._data_seconds,
            "compute_seconds": self._compute_seconds,
            "data_fraction": data_fraction,
        }
        if total > 0 and data_fraction >= 0.4:
            self._emit(
                Issue(
                    step=None,
                    code="dataloader_bottleneck",
                    severity="warning",
                    message=(
                        f"{data_fraction*100:.1f}% of measured time was spent "
                        f"waiting on data loading, not compute. The GPU is likely "
                        f"idling between batches. Typical fixes: more DataLoader "
                        f"workers, `pin_memory=True`, or moving heavy "
                        f"preprocessing/augmentation off the critical path."
                    ),
                    data=report,
                )
            )

        if self._gpu_timer_samples > 0:
            gpu_idle_fraction = (
                1.0 - (self._gpu_busy_seconds / self._gpu_block_wall_seconds)
                if self._gpu_block_wall_seconds > 0
                else 0.0
            )
            gpu_idle_fraction = max(0.0, gpu_idle_fraction)
            report["gpu_block_wall_seconds"] = self._gpu_block_wall_seconds
            report["gpu_busy_seconds"] = self._gpu_busy_seconds
            report["gpu_idle_fraction"] = gpu_idle_fraction
            report["gpu_timer_samples"] = self._gpu_timer_samples
            if gpu_idle_fraction >= 0.2:
                self._emit(
                    Issue(
                        step=None,
                        code="gpu_launch_stall",
                        severity="warning",
                        message=(
                            f"Inside code wrapped in gpu_timer(), the GPU was only "
                            f"actually executing for {(1-gpu_idle_fraction)*100:.1f}% "
                            f"of the wall-clock time spent in that block "
                            f"(measured via torch.cuda.Event, not nvidia-smi%). This "
                            f"is time the GPU spent idle between/inside kernel "
                            f"launches that a plain utilization-percent reading "
                            f"would not show. Common causes: small ops issued one "
                            f"at a time (Python/launch overhead per kernel is fixed "
                            f"cost, so many small ops pay it repeatedly), an "
                            f"unintended `.item()`/`.cpu()`/`print()` forcing a "
                            f"host-device sync mid-step, or CPU-side work "
                            f"(indexing, control flow) sitting on the critical path "
                            f"between kernel launches. This number is a measurement "
                            f"of where time went, not a diagnosis of which of these "
                            f"applies here — check with a proper profiler "
                            f"(`torch.profiler`) for that."
                        ),
                        data={
                            "gpu_idle_fraction": gpu_idle_fraction,
                            "gpu_busy_seconds": self._gpu_busy_seconds,
                            "gpu_block_wall_seconds": self._gpu_block_wall_seconds,
                        },
                    )
                )
        return report

    def summary(self) -> List[Issue]:
        """All issues recorded so far, in order."""
        return list(self.issues)

    # -- phase / progress tracking -------------------------------------
    def record_step_time(self, step: int, total_steps: Optional[int] = None) -> PhaseStatus:
        """Call once per training step (e.g. right after
        `optimizer.step()`, once per iteration of your loop) to track
        real wall-clock step timing and answer three concrete
        questions: is training still in its noisy startup window or
        has it settled into steady state, is the current step an
        outlier (stall) relative to that steady state, and — if you
        pass `total_steps` — how much wall-clock time is left at the
        current steady-state rate.

        How phase classification works
        -------------------------------
        - `warmup`: the first `self.warmup_steps` calls (see
          `TrainingMonitor.__init__`'s `warmup_steps`/
          `expected_param_count` — a heuristic, not a measurement; see
          `_default_warmup_steps`'s docstring). These steps are
          recorded for `elapsed_seconds`/`steps_recorded` but
          deliberately excluded from `steady_state_avg_seconds` — cuDNN
          benchmark autotuning, `torch.compile` first-compile, and CUDA
          allocator cache warmup make these steps legitimately slower
          without anything being wrong.
        - `stalled`: (only possible after warmup, and only once at
          least `self.stall_min_samples` steady-state steps have been
          recorded — a single early reading isn't a stable-enough
          baseline to call anything an outlier against) the CURRENT
          step's time is more than `self.stall_factor`x the running
          steady-state average. Typical causes: a DataLoader hiccup, a
          checkpoint save landing on the critical path, thermal
          throttling, or another process on the machine (see
          `WinCore.thermal`/`WinCore.spec` for signals you can feed
          into `record_signal` to correlate against this). A stalled
          step's time is NOT folded into the steady-state average
          (same reasoning as excluding warmup: one bad step shouldn't
          drag the baseline used to judge the next one), but IS still
          counted in `steps_recorded`/`elapsed_seconds`, and emits a
          `step_stall` warning `Issue`.
        - `steady_state`: anything else — the common case for a
          healthy run.

        ETA (`eta_seconds` in the returned `PhaseStatus`)
        -----------------------------------------------------
        Only computed once `steady_state_avg_seconds` is known (i.e.
        after warmup AND after at least one non-stalled post-warmup
        step) AND `total_steps` was passed to THIS call. Computed as
        `(total_steps - step) * steady_state_avg_seconds` — a simple
        linear projection at the current steady-state rate, not a
        prediction that accounts for a learning-rate schedule changing
        speed, a dataset epoch boundary, or anything else that could
        change per-step cost later in the run. Treat it as "if the next
        N steps run like the last several did", not a guarantee.

        Args:
            step: the current (global) training step number. Used only
                for the returned `PhaseStatus.step` and for the ETA
                calculation (`total_steps - step`) — this method's own
                phase/stall logic is based on call *order* and elapsed
                wall-clock time, not on `step` needing to increment by
                exactly 1 each call.
            total_steps: if given, enables `eta_seconds` in the
                returned status (see above). `None` (default) leaves
                `eta_seconds` as `None` — this method never guesses a
                total on its own.

        Returns a `PhaseStatus` snapshot (see its own docstring for
        every field). Also appends to `self.issues` / calls
        `self.on_issue` when `phase == "stalled"`, same as every other
        `record_*` method's issue-emission convention in this class.
        """
        now = time.perf_counter()
        if self._step_time_start is None:
            self._step_time_start = now
            self._last_step_ts = now

        step_seconds = now - self._last_step_ts
        self._last_step_ts = now
        self._steps_recorded += 1

        in_warmup = self._steps_recorded <= self.warmup_steps
        steady_avg = (
            sum(self._steady_state_step_seconds) / len(self._steady_state_step_seconds)
            if self._steady_state_step_seconds
            else None
        )

        if in_warmup:
            phase = "warmup"
            reason = (
                f"Step {self._steps_recorded} of {self.warmup_steps} warmup steps "
                f"(cuDNN benchmark / kernel JIT / allocator cache warmup expected "
                f"here) — not counted toward the steady-state average."
            )
        elif (
            steady_avg is not None
            and len(self._steady_state_step_seconds) >= self.stall_min_samples
            and step_seconds > steady_avg * self.stall_factor
        ):
            phase = "stalled"
            reason = (
                f"This step took {step_seconds:.3f}s, {step_seconds/steady_avg:.1f}x "
                f"the steady-state average ({steady_avg:.3f}s) — likely a "
                f"DataLoader hiccup, checkpoint I/O on the critical path, thermal "
                f"throttling, or another process competing for the GPU/CPU. Not "
                f"folded into the steady-state average."
            )
        else:
            phase = "steady_state"
            self._steady_state_step_seconds.append(step_seconds)
            steady_avg = sum(self._steady_state_step_seconds) / len(self._steady_state_step_seconds)
            reason = (
                f"Steady state — running average {steady_avg:.3f}s/step over "
                f"{len(self._steady_state_step_seconds)} sample(s)."
            )

        eta_seconds = None
        if total_steps is not None and steady_avg is not None:
            remaining = max(0, total_steps - step)
            eta_seconds = remaining * steady_avg

        status = PhaseStatus(
            phase=phase,
            step=step,
            steps_recorded=self._steps_recorded,
            elapsed_seconds=now - self._step_time_start,
            last_step_seconds=step_seconds,
            steady_state_avg_seconds=steady_avg,
            steps_per_second=(1.0 / steady_avg) if steady_avg else None,
            eta_seconds=eta_seconds,
            reason=reason,
        )

        if phase == "stalled":
            self._emit(
                Issue(
                    step=step,
                    code="step_stall",
                    severity="warning",
                    message=(
                        f"Step {step} stalled: {reason}"
                    ),
                    data={
                        "step_seconds": step_seconds,
                        "steady_state_avg_seconds": steady_avg,
                        "factor": step_seconds / steady_avg if steady_avg else None,
                    },
                )
            )

        return status


# -- background-level (hook-based) NaN/Inf detection ---------------------
#
# `record_loss` above catches NaN/Inf once it has already propagated all
# the way to the scalar loss -- by then you know *that* something broke,
# but not *where* inside the model it started, or which one of N steps
# between two loss checks introduced it. `attach_nan_guards` instead
# hooks every submodule's forward (and optionally backward) pass, so the
# very first layer to produce a NaN/Inf output is caught at the moment
# it happens, not inferred after the fact from a suspicious loss value.
#
# This does add per-layer overhead (one `torch.isfinite` reduction per
# hooked module, per forward/backward call), so it is meant for
# debugging a suspected instability, not for permanent every-step
# production use on every training run -- `enabled` lets you toggle it
# off without removing the hooks, and `detach()` removes them entirely.


class NaNGuardHandle:
    """Returned by `attach_nan_guards`. Call `.detach()` to remove every
    hook it registered, or toggle `.enabled = False` to pause detection
    (e.g. after diagnosing the issue) without re-registering hooks later."""

    def __init__(self, handles: List[Any]):
        self._handles = handles
        self.enabled = True

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self) -> "NaNGuardHandle":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.detach()
        return False


def attach_nan_guards(
    model,
    on_issue: Optional[Callable[[Issue], None]] = None,
    check_forward: bool = True,
    check_backward: bool = True,
    raise_on_detect: bool = False,
) -> NaNGuardHandle:
    """Register forward/backward hooks on every named submodule of
    `model` that flag the *first* module whose output (forward) or
    input gradient (backward) contains a NaN/Inf -- pinpointing where
    a numerical problem originates, instead of only seeing it after it
    has already reached the loss.

    Args:
        model: an `nn.Module`. Hooks are attached to every submodule
            returned by `model.named_modules()`, including `model`
            itself.
        on_issue: called with an `Issue` (code="module_output_nan" or
            "module_grad_nan") the first time a hooked module produces
            a non-finite forward output or backward gradient.
        check_forward: hook `forward` outputs.
        check_backward: hook the input gradient flowing into each
            module during backward (via `register_full_backward_hook`).
        raise_on_detect: if True, raise `FloatingPointError` immediately
            instead of (or in addition to) calling `on_issue` -- use
            this when you want training to stop hard the instant a NaN
            appears, rather than continuing with the callback as the
            only signal.

    Returns:
        A `NaNGuardHandle` — call `.detach()` when done (or use it as a
        context manager) to remove the hooks; leaving them attached
        permanently costs a finite-check per hooked module per
        forward/backward call, which is real but usually small overhead
        relative to the matmul/conv work in each layer.

    Only checks the *first* tensor-shaped output/grad of each module for
    simplicity — multi-output modules (e.g. LSTM returning `(out,
    (h, c))`) are walked recursively through tuples/lists so every
    tensor leaf is checked, not just position 0.
    """
    import torch

    handles: List[Any] = []
    guard_state = {"triggered": False}

    def _first_nonfinite(value) -> Optional["torch.Tensor"]:
        """Recurse through tuples/lists/dicts of tensors and return the
        first non-finite tensor found, or None."""
        if isinstance(value, torch.Tensor):
            if value.is_floating_point() and not torch.isfinite(value).all():
                return value
            return None
        if isinstance(value, (list, tuple)):
            for item in value:
                found = _first_nonfinite(item)
                if found is not None:
                    return found
            return None
        if isinstance(value, dict):
            for item in value.values():
                found = _first_nonfinite(item)
                if found is not None:
                    return found
            return None
        return None

    def _emit(guard: NaNGuardHandle, name: str, code: str, tensor) -> None:
        if not guard.enabled or guard_state["triggered"]:
            return
        guard_state["triggered"] = True
        kind = "Inf" if torch.isinf(tensor).any() and not torch.isnan(tensor).any() else "NaN"
        stage = "output" if code == "module_output_nan" else "input gradient"
        issue = Issue(
            step=None,
            code=code,
            severity="critical",
            message=(
                f"Module '{name}' ({tensor.dtype}, shape {tuple(tensor.shape)}) "
                f"produced a non-finite ({kind}) {stage} — this is the first "
                f"module where the problem was detected, walking the model in "
                f"execution order, so earlier layers were still finite at this "
                f"point. Typical causes: a division/log/sqrt of zero or a "
                f"negative value inside this specific layer, an unstable "
                f"activation for the current input scale, or (in backward) a "
                f"loss scale too high for mixed precision."
            ),
            data={"module": name, "kind": kind, "stage": stage},
        )
        if on_issue is not None:
            on_issue(issue)
        if raise_on_detect:
            raise FloatingPointError(issue.message)

    guard = NaNGuardHandle(handles)

    for name, module in model.named_modules():
        display_name = name or model.__class__.__name__

        if check_forward:

            def _fwd_hook(mod, inputs, output, _name=display_name):
                bad = _first_nonfinite(output)
                if bad is not None:
                    _emit(guard, _name, "module_output_nan", bad)

            handles.append(module.register_forward_hook(_fwd_hook))

        if check_backward and hasattr(module, "register_full_backward_hook"):

            def _bwd_hook(mod, grad_input, grad_output, _name=display_name):
                bad = _first_nonfinite(grad_input)
                if bad is not None:
                    _emit(guard, _name, "module_grad_nan", bad)

            handles.append(module.register_full_backward_hook(_bwd_hook))

    return guard
