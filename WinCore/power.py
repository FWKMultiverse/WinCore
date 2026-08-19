"""
WinCore.power -- two real Windows-specific pain points for long AI
training runs that no other module here touches: the OS putting the
whole machine to sleep mid-run, and the GPU driver silently killing a
long-running CUDA kernel launch.

Why these matter specifically for AI training on Windows
-----------------------------------------------------------
1. Sleep/display-off during a multi-hour run. Windows' own power plan
   (even "Balanced", not just "Power saver") suspends the machine
   after its configured idle timeout REGARDLESS of GPU/CPU load if
   there's no user input (mouse/keyboard) for that long -- a training
   loop has no UI and produces exactly zero of the "activity" Windows'
   idle timer looks for. The practical symptom: start a run before
   bed, wake up to a suspended machine and hours of lost progress --
   with no crash, no traceback, no log entry explaining why, since the
   process is still literally there, just frozen along with everything
   else. It looks like a hang, not a sleeping machine.

2. TDR (Timeout Detection and Recovery). Windows' GPU driver watchdog
   kills and resets a CUDA kernel that runs longer than ~2 seconds by
   default (`TdrDelay` in the registry) -- a safety mechanism meant to
   recover a frozen desktop from a buggy DISPLAY driver, applied
   uniformly to compute workloads too, with no CUDA-awareness. This
   bites custom CUDA kernels specifically -- exactly the kind
   `WinCore.kernels` ships (`fused_bias_gelu`) -- and shows up as
   `RuntimeError: CUDA error: unspecified launch failure` with nothing
   in that message pointing at a 2-second Windows timer as the actual
   cause.

Neither of these can be silently auto-fixed from user-mode Python:
sleep prevention needs an actively held request for as long as
training runs (`prevent_sleep()` below gives you that); TDR needs an
admin registry edit plus a reboot, so this module can only detect and
clearly explain it (`check_tdr_risk()`), not change it on your behalf.
"""
from __future__ import annotations

import platform
from dataclasses import dataclass
from typing import Optional


class PowerError(RuntimeError):
    """Raised by `prevent_sleep()`'s `start()` if the real Windows API
    call itself reports failure (rare -- typically means something
    else is already badly wrong with the process)."""


_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001
_ES_DISPLAY_REQUIRED = 0x00000002


class _SleepPreventer:
    """Context manager returned by `prevent_sleep()`. Also directly
    callable as `.start()` / `.stop()` for callers that can't use a
    `with` block -- e.g. a Jupyter notebook where the run spans many
    cells, or a long-lived training daemon started and stopped from
    separate code paths."""

    def __init__(self, keep_display_on: bool):
        self.keep_display_on = keep_display_on
        self._active = False

    def start(self) -> bool:
        """Issue the request. Returns True if applied (Windows only,
        real API call succeeded), False on any other platform -- a
        deliberate, silent no-op (see module docstring for why this is
        a Windows-specific concern in the first place)."""
        if platform.system() != "Windows":
            return False
        import ctypes

        flags = _ES_CONTINUOUS | _ES_SYSTEM_REQUIRED
        if self.keep_display_on:
            flags |= _ES_DISPLAY_REQUIRED
        result = ctypes.windll.kernel32.SetThreadExecutionState(flags)
        if result == 0:
            raise PowerError(
                "SetThreadExecutionState failed to apply the sleep-prevention "
                f"request (GetLastError={ctypes.get_last_error()})"
            )
        self._active = True
        return True

    def stop(self) -> None:
        """Release the request, letting Windows' normal power plan
        (including sleep) resume applying as configured. Safe to call
        even if `start()` was never called or failed -- this is
        cleanup, not a paired lock that must balance."""
        if platform.system() != "Windows" or not self._active:
            return
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS)
        self._active = False

    def __enter__(self) -> "_SleepPreventer":
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()


def prevent_sleep(keep_display_on: bool = False) -> _SleepPreventer:
    """Stop Windows from suspending the machine for as long as this is
    held -- use as a context manager around a training run:

        with WinCore.power.prevent_sleep():
            for epoch in range(num_epochs):
                train_one_epoch(...)

    A real OS call (`SetThreadExecutionState`), not a background thread
    jiggling the mouse or any such workaround -- this is the documented
    Win32 mechanism apps like video players and installers use to stop
    the system (and optionally display) from sleeping during long-
    running, non-interactive work. Per Windows' own semantics for this
    API, the request only lasts as long as the calling THREAD stays
    alive -- for a typical single-process, single-main-thread training
    script that's exactly the process's lifetime, which is what you
    want. If your training loop instead runs on a background thread
    that can exit before the main thread does, hold this on whichever
    thread's lifetime should actually determine "still training".

    Args:
        keep_display_on: if True, also prevents the DISPLAY from
            turning off (`ES_DISPLAY_REQUIRED`), not just system sleep.
            Off by default -- for unattended training a black screen is
            fine and saves power; the machine itself staying awake
            (`ES_SYSTEM_REQUIRED`, always applied) is what actually
            matters for the run to keep executing.

    On non-Windows platforms this is a no-op context manager (no
    error) -- see the module docstring for why this is specifically a
    Windows concern rather than something every OS needs guarding
    against equally.

    Raises `PowerError` if this IS Windows and the real API call
    itself reports failure.
    """
    return _SleepPreventer(keep_display_on=keep_display_on)


@dataclass
class TdrReport:
    """Result of `check_tdr_risk()`."""

    platform_is_windows: bool
    tdr_delay_seconds: Optional[int]  # None only when platform_is_windows is False
    at_default_risk_level: bool
    message: str


def check_tdr_risk() -> TdrReport:
    r"""Read Windows' GPU driver watchdog timeout (`TdrDelay`, under
    `HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers`) and report
    whether a single CUDA kernel launch has only the OS default ~2
    seconds before Windows kills and resets the GPU driver.

    This is read-only diagnostics, not a fix -- `TdrDelay` genuinely
    requires an admin registry edit AND a reboot to change, and doing
    that from an unprivileged training script would need to assume
    admin rights this package has no business assuming it has. What
    this DOES give you: an honest, specific answer to "why did my
    custom kernel just throw `CUDA error: unspecified launch failure`
    on a clean input, with no OOM and no NaN in sight" -- one of the
    most confusing failure modes for someone writing or compiling their
    own CUDA kernels on Windows (exactly the situation
    `WinCore.kernels` puts a user in), since that error message alone
    gives zero indication that a Windows-specific 2-second timer, not a
    bug in the kernel, caused it.

    Returns a `TdrReport`:
      - `tdr_delay_seconds`: the configured value, or the OS default of
        2 if the registry value has never been explicitly set -- a
        missing key means "using the 2s default", not "no timeout",
        and is the actual common case, since most machines have never
        had this touched.
      - `at_default_risk_level`: True whenever `tdr_delay_seconds <= 2`
        (the low-headroom default), whether that's because the key is
        missing or explicitly set that low.
      - `message`: a one-paragraph, copy-pasteable explanation, plus
        the exact registry path/value to raise it and the reminder
        that it needs a reboot to take effect.

    On non-Windows, `platform_is_windows` is False and everything else
    is a fixed, honest "not applicable" -- there's no directly
    equivalent watchdog on Linux's NVIDIA driver for compute workloads
    by default.
    """
    if platform.system() != "Windows":
        return TdrReport(
            platform_is_windows=False,
            tdr_delay_seconds=None,
            at_default_risk_level=False,
            message=(
                "TDR is a Windows-specific GPU driver watchdog; not applicable "
                "on this platform."
            ),
        )

    tdr_delay_seconds = 2  # Windows' own default when the value has never been set
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\GraphicsDrivers",
        ) as key:
            tdr_delay_seconds, _ = winreg.QueryValueEx(key, "TdrDelay")
    except OSError:
        # Key or value doesn't exist -- means "never configured", i.e.
        # still on Windows' 2-second default, not "no timeout".
        pass

    tdr_delay_seconds = int(tdr_delay_seconds)
    at_risk = tdr_delay_seconds <= 2

    if at_risk:
        message = (
            f"TdrDelay is {tdr_delay_seconds}s (Windows' own default, or explicitly "
            "set that low) -- any single CUDA kernel launch running longer than "
            "this gets killed by Windows' driver watchdog and resets the GPU, "
            "surfacing in PyTorch as `CUDA error: unspecified launch failure` with "
            "no indication the real cause is this timer rather than your kernel or "
            "model code. This mainly bites large single-kernel-launch ops (a big "
            "custom kernel launch, a huge single matmul/conv, or a debugger "
            "breakpoint hit mid-kernel) -- most ordinary training steps finish well "
            "under 2s per kernel launch and never hit this at all. If you ARE "
            "hitting it: raise TdrDelay (needs Administrator + a reboot to take "
            r"effect) at HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers"
            "\\TdrDelay (create it as a DWORD if missing; value = desired seconds, "
            "e.g. 10)."
        )
    else:
        message = (
            f"TdrDelay is {tdr_delay_seconds}s, above the 2s default -- single "
            "kernel launches have more headroom before Windows' driver watchdog "
            "would kill and reset the GPU."
        )

    return TdrReport(
        platform_is_windows=True,
        tdr_delay_seconds=tdr_delay_seconds,
        at_default_risk_level=at_risk,
        message=message,
    )
