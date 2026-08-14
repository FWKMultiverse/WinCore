"""
System spec detection — RAM, VRAM, GPU name, and a minimum-requirement check.

Honesty about scope
--------------------
This module reads what the OS/driver will actually report and nothing
more:
  - Total/available system RAM: via `psutil` (optional dependency).
  - GPU name + total/free VRAM: via `pynvml` for NVIDIA cards (optional
    dependency), or via `torch.cuda` if torch is already installed and
    CUDA-enabled.
  - AMD cards: name only, best-effort, via Windows WMI (`wmi` package,
    optional, Windows-only). Free/total VRAM for AMD is NOT exposed by
    any pure-Python cross-vendor API, so `vram_total_gb` is left `None`
    for AMD unless the caller supplies it manually.

There is no reliable, portable way to detect a CPU's *generation*
(e.g. "i5-9400f or newer") from Python alone — CPU name strings are not
standardized enough to parse safely. `meets_minimum()` therefore checks
what can actually be measured: logical core/thread count and VRAM, not
a specific model/generation. If you need a hard model-generation gate,
pass an explicit allow-list of substrings via `cpu_name_contains`.
"""
from __future__ import annotations

import platform
from dataclasses import dataclass, field
from typing import Iterable, List, Optional


@dataclass
class GPUInfo:
    name: Optional[str] = None
    vendor: Optional[str] = None  # "nvidia" | "amd" | "unknown"
    vram_total_gb: Optional[float] = None
    vram_free_gb: Optional[float] = None
    temperature_c: Optional[float] = None
    source: Optional[str] = None  # which backend supplied this info


@dataclass
class SystemSpec:
    os_name: str
    logical_threads: int
    physical_cores: Optional[int]
    ram_total_gb: Optional[float]
    ram_available_gb: Optional[float]
    gpus: List[GPUInfo] = field(default_factory=list)


def _ram_info():
    try:
        import psutil
    except ImportError:
        return None, None
    vm = psutil.virtual_memory()
    return round(vm.total / (1024**3), 2), round(vm.available / (1024**3), 2)


def _cpu_counts():
    logical = None
    physical = None
    try:
        import psutil

        logical = psutil.cpu_count(logical=True)
        physical = psutil.cpu_count(logical=False)
    except ImportError:
        pass
    if logical is None:
        import os

        logical = os.cpu_count() or 1
    return logical, physical


def _nvidia_gpus_via_pynvml() -> List[GPUInfo]:
    gpus: List[GPUInfo] = []
    try:
        import pynvml

        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", "ignore")
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                temp = None
            gpus.append(
                GPUInfo(
                    name=name,
                    vendor="nvidia",
                    vram_total_gb=round(mem.total / (1024**3), 2),
                    vram_free_gb=round(mem.free / (1024**3), 2),
                    temperature_c=temp,
                    source="pynvml",
                )
            )
        pynvml.nvmlShutdown()
    except Exception:
        return []
    return gpus


def _gpus_via_torch() -> List[GPUInfo]:
    gpus: List[GPUInfo] = []
    try:
        import torch

        if not torch.cuda.is_available():
            return []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            free_b, total_b = torch.cuda.mem_get_info(i)
            gpus.append(
                GPUInfo(
                    name=props.name,
                    vendor="nvidia" if "nvidia" in props.name.lower() or True else "unknown",
                    vram_total_gb=round(total_b / (1024**3), 2),
                    vram_free_gb=round(free_b / (1024**3), 2),
                    source="torch.cuda",
                )
            )
    except Exception:
        return []
    return gpus


def _gpu_names_via_wmi() -> List[GPUInfo]:
    """Best-effort GPU *name* lookup on Windows for any vendor (no VRAM)."""
    if platform.system() != "Windows":
        return []
    try:
        import wmi  # optional, Windows-only

        conn = wmi.WMI()
        gpus = []
        for gpu in conn.Win32_VideoController():
            name = getattr(gpu, "Name", None)
            if not name:
                continue
            vendor = "unknown"
            lname = name.lower()
            if "nvidia" in lname or "geforce" in lname or "rtx" in lname or "gtx" in lname:
                vendor = "nvidia"
            elif "amd" in lname or "radeon" in lname:
                vendor = "amd"
            gpus.append(GPUInfo(name=name, vendor=vendor, source="wmi"))
        return gpus
    except Exception:
        return []


def get_gpu_temperature(index: int = 0) -> Optional[float]:
    """Read the current temperature (°C) of GPU `index`, in one direct
    call — cheaper than `get_system_spec()` when polling repeatedly
    inside a training loop.

    This calls NVIDIA's own `pynvml.nvmlDeviceGetTemperature` directly —
    it is NOT a custom sensor implementation, just a thin pass-through
    to the vendor's own helper. Returns `None` if `pynvml` isn't
    installed, there's no NVIDIA GPU, or the query fails for any
    reason (e.g. driver doesn't expose it) — never raises, since a
    training loop calling this every N steps shouldn't crash over a
    monitoring read.
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        pynvml.nvmlShutdown()
        return float(temp)
    except Exception:
        return None


def get_gpus() -> List[GPUInfo]:
    """Return whatever GPU info can actually be measured on this machine.

    Tries, in order: pynvml (best VRAM detail for NVIDIA) -> torch.cuda
    (if torch is installed) -> WMI on Windows (name only, any vendor).
    Returns an empty list if none of these are available/applicable —
    this is not an error, just "nothing detectable without an optional
    dependency installed".
    """
    gpus = _nvidia_gpus_via_pynvml()
    if gpus:
        return gpus
    gpus = _gpus_via_torch()
    if gpus:
        return gpus
    return _gpu_names_via_wmi()


def get_system_spec() -> SystemSpec:
    logical, physical = _cpu_counts()
    ram_total, ram_avail = _ram_info()
    return SystemSpec(
        os_name=platform.system(),
        logical_threads=logical,
        physical_cores=physical,
        ram_total_gb=ram_total,
        ram_available_gb=ram_avail,
        gpus=get_gpus(),
    )


@dataclass
class MinimumCheckResult:
    ok: bool
    reasons: List[str]
    spec: SystemSpec


def meets_minimum(
    min_logical_threads: int = 1,
    min_vram_gb: Optional[float] = None,
    min_ram_gb: Optional[float] = None,
    cpu_name_contains: Optional[Iterable[str]] = None,
    spec: Optional[SystemSpec] = None,
) -> MinimumCheckResult:
    """Check the current machine against caller-supplied minimums.

    Only checks what can actually be measured (see module docstring for
    why CPU generation isn't one of them). `cpu_name_contains` lets you
    add a soft allow-list check (e.g. `["i5-9", "i5-1", "i7", "i9",
    "ryzen 5", "ryzen 7"]`) against `platform.processor()`, but that
    string is OS/vendor-dependent and not guaranteed to be present or
    accurate — treat it as advisory, not authoritative.
    """
    spec = spec or get_system_spec()
    reasons: List[str] = []

    if spec.logical_threads < min_logical_threads:
        reasons.append(
            f"logical threads {spec.logical_threads} < required {min_logical_threads}"
        )

    if min_ram_gb is not None:
        if spec.ram_total_gb is None:
            reasons.append("RAM size unknown (install 'psutil' to check this)")
        elif spec.ram_total_gb < min_ram_gb:
            reasons.append(f"RAM {spec.ram_total_gb}GB < required {min_ram_gb}GB")

    if min_vram_gb is not None:
        best_vram = max((g.vram_total_gb or 0 for g in spec.gpus), default=None)
        if not spec.gpus:
            reasons.append("no GPU detected (install 'nvidia-ml-py' for NVIDIA VRAM checks)")
        elif best_vram is None or best_vram == 0:
            reasons.append(
                "GPU detected but VRAM size unknown for this vendor "
                "(pynvml only covers NVIDIA)"
            )
        elif best_vram < min_vram_gb:
            reasons.append(f"VRAM {best_vram}GB < required {min_vram_gb}GB")

    if cpu_name_contains:
        cpu_name = (platform.processor() or "").lower()
        if not any(sub.lower() in cpu_name for sub in cpu_name_contains):
            reasons.append(
                f"CPU name '{platform.processor()}' did not match any of "
                f"{list(cpu_name_contains)} (advisory check only)"
            )

    return MinimumCheckResult(ok=not reasons, reasons=reasons, spec=spec)
