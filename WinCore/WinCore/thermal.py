"""
Thermal guard for training loops — reads temperature via NVIDIA's own
pynvml, pauses the training loop (software-level) if it's too hot.

Scope, stated plainly
----------------------
This does NOT control the GPU's fan curve, power limit, or clocks —
Python has no portable access to that (it lives in the driver/BIOS/
vendor overlay software, e.g. NVIDIA's own throttling when a card hits
its thermal limit, which already happens regardless of this module).

What this DOES do: read the temperature NVIDIA's driver already reports
(via `spec.get_gpu_temperature`, itself a thin call to
`pynvml.nvmlDeviceGetTemperature`), and if it's above your threshold,
sleep the training loop for a bit before the next step. Giving the GPU
idle time lets its *existing* fan curve/driver cool it down faster than
it would under continuous full load — this is a real, if blunt,
software-side mitigation (some training scripts already do this
manually with a `time.sleep()` in a temperature-check callback). It is
not a substitute for adequate case airflow or a fan curve that's
actually adequate for your workload.

Graduated response, not one fixed pause
-----------------------------------------
The first version of this guard always slept for exactly
`pause_seconds` the instant temperature crossed `threshold_c`, every
single time it was over -- the same fixed-severity response whether
it's 1 degree over or way over, and whether this is the first check
over threshold or the fiftieth in a row. That's blunt in both
directions: doesn't ease off (settle for a shorter pause) once it's
clear a small dip already helped, and doesn't escalate if temperature
keeps climbing anyway.

Now the pause grows geometrically (`pause_seconds * backoff_factor **
consecutive_overheat_checks`, capped at `max_pause_seconds`) the longer
temperature has stayed above `threshold_c` on consecutive `.check()`
calls, and resets back to the base `pause_seconds` the moment a check
comes back under threshold again -- so a brief, easily-recovered spike
gets a short pause, and sustained overheating gets progressively more
aggressive cooldown instead of repeating the same fixed pause
indefinitely.

`critical_threshold_c` (optional, above `threshold_c`) marks a more
severe line -- if crossed, `on_critical(event)` fires BEFORE the sleep,
specifically so a caller can save a checkpoint (e.g. via
`WinCore.io.atomic_torch_save`) while there's still time, rather than
finding out mid-write that the process needs to stop. This module does
NOT decide to abort training on your behalf at any temperature --
`on_critical` is a hook for you to act on (checkpoint, log, alert,
raise your own exception to stop the loop, or nothing at all), not a
built-in kill switch.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

from .spec import get_gpu_temperature


@dataclass
class ThermalEvent:
    temperature_c: float
    threshold_c: float
    paused_seconds: float
    critical: bool = False  # True if this reading also crossed critical_threshold_c


class ThermalGuard:
    """Call `.check()` periodically (e.g. every N training steps).

    Example:
        guard = WinCore.thermal.ThermalGuard(threshold_c=83)
        for step, batch in enumerate(loader):
            train_step(batch)
            if step % 20 == 0:
                guard.check()

        # with graduated backoff and a critical auto-save hook:
        guard = WinCore.thermal.ThermalGuard(
            threshold_c=83, critical_threshold_c=90,
            on_critical=lambda e: WinCore.io.atomic_torch_save(
                model.state_dict(), "emergency_checkpoint.pt"
            ),
        )
    """

    def __init__(
        self,
        threshold_c: float = 83.0,
        pause_seconds: float = 5.0,
        gpu_index: int = 0,
        on_pause: Optional[Callable[[ThermalEvent], None]] = None,
        monitor=None,
        critical_threshold_c: Optional[float] = None,
        on_critical: Optional[Callable[[ThermalEvent], None]] = None,
        max_pause_seconds: float = 30.0,
        backoff_factor: float = 1.5,
    ):
        self.threshold_c = threshold_c
        self.pause_seconds = pause_seconds
        self.gpu_index = gpu_index
        self.on_pause = on_pause
        self.monitor = monitor  # optional WinCore.diagnostics.TrainingMonitor
        self.critical_threshold_c = critical_threshold_c
        self.on_critical = on_critical
        self.max_pause_seconds = max_pause_seconds
        self.backoff_factor = backoff_factor
        self.last_temperature_c: Optional[float] = None
        self.total_paused_seconds: float = 0.0
        self._consecutive_overheat_checks = 0

    def check(self, step: Optional[int] = None) -> Optional[ThermalEvent]:
        """Read temperature once; sleep an escalating (then resetting)
        pause if over `threshold_c` -- see the module docstring's
        "Graduated response" section for exactly how the pause duration
        is computed. Returns a `ThermalEvent` if it paused, else `None`.
        Silently does nothing (returns `None`) if temperature can't be
        read (no pynvml / no NVIDIA GPU) — this guard degrades to a
        no-op rather than breaking a training loop that doesn't have
        it available.

        If constructed with `monitor=some_training_monitor`, every
        reading (not just pauses) is fed to
        `monitor.record_signal(step, "gpu_temp_c", temp)` -- pass the
        current step here so a later loss/gradient Issue near this step
        gets annotated with the temperature at the time. Optional: skip
        `monitor`/`step` entirely and this behaves exactly as before.

        If `critical_threshold_c` is set and this reading meets or
        exceeds it, `on_critical(event)` fires BEFORE the sleep -- see
        the module docstring."""
        temp = get_gpu_temperature(self.gpu_index)
        self.last_temperature_c = temp
        if self.monitor is not None and temp is not None:
            self.monitor.record_signal(step, "gpu_temp_c", temp)
        if temp is None or temp < self.threshold_c:
            self._consecutive_overheat_checks = 0
            return None

        self._consecutive_overheat_checks += 1
        pause = min(
            self.pause_seconds * (self.backoff_factor ** (self._consecutive_overheat_checks - 1)),
            self.max_pause_seconds,
        )
        critical = self.critical_threshold_c is not None and temp >= self.critical_threshold_c

        event = ThermalEvent(
            temperature_c=temp, threshold_c=self.threshold_c, paused_seconds=pause, critical=critical
        )

        if critical and self.on_critical is not None:
            # fires BEFORE the sleep, deliberately -- the whole point is
            # giving a caller (e.g. a checkpoint save) time to run while
            # the situation is still recoverable, not after.
            self.on_critical(event)

        time.sleep(pause)
        self.total_paused_seconds += pause
        if self.monitor is not None:
            note = f"paused {pause:.1f}s at {temp}C" + (" [CRITICAL]" if critical else "")
            self.monitor.record_signal(step, "thermal_pause", pause, note=note)
        if self.on_pause is not None:
            self.on_pause(event)
        return event
