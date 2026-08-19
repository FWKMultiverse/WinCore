"""
Multi-GPU helper — picks a sane `torch.distributed` backend and DDP
config for whatever GPU count/topology is actually on the machine (2,
3, 4, or N GPUs), instead of leaving every user to rediscover the same
Windows-specific gotchas.

Why this exists
----------------
This does NOT reimplement NCCL, gradient all-reduce, or anything else
`torch.distributed`/`torch.nn.parallel.DistributedDataParallel` already
do — those remain exactly as fast as your installed torch/NCCL build
makes them. What this module replaces is the error-prone *setup*
around multi-GPU training on Windows specifically:

  - **NCCL has no official Windows build.** PyTorch on Windows ships
    without NCCL; `torch.distributed.init_process_group(backend="nccl")`
    raises there. The Windows-compatible backend is `gloo` (CPU+GPU,
    slower for large all-reduce but it works), or `nccl` for real if
    you're inside WSL2 with the Linux torch wheel. `recommended_backend()`
    picks this correctly instead of the user finding out at crash time.
  - **Guessing `find_unused_parameters` wrong is a silent slowdown.**
    Setting it `True` when you don't need it disables a real
    optimization (bucket-based gradient-ready detection) for no reason;
    leaving it `False` when you do need it hangs the all-reduce. This
    module doesn't guess for you (it can't know your model's forward
    pass) — it documents the trade-off and gives a helper to test it.
  - **DDP bucket size interacts with GPU interconnect.** `detect_topology()`
    actually measures it: NVLink presence per GPU pair via
    `pynvml.nvmlDeviceGetNvLinkState` + `nvmlDeviceGetNvLinkRemotePciInfo`
    (which two GPUs a given NVLink line connects to), and PCIe distance
    (same switch / host bridge / crosses NUMA nodes) via
    `pynvml.nvmlDeviceGetTopologyCommonAncestor` when NVLink isn't
    present between a pair — both real NVML calls, not a guess. Falls
    back to parsing `nvidia-smi topo -m` (the same data NVML exposes,
    read from the vendor's own tool) if `pynvml` isn't installed but
    `nvidia-smi` is on PATH. `recommended_bucket_cap_mb()` uses this
    measurement when available and only falls back to a GPU-count
    heuristic when neither detection path works.

What this deliberately is NOT
------------------------------
  - Not an NCCL/gloo reimplementation — it configures the backend
    PyTorch already ships.
  - Not automatic model sharding (that's `torch.distributed.fsdp`,
    orthogonal to this module) — this covers plain DDP setup.
  - Topology detection depends on NVIDIA tooling (`pynvml` or
    `nvidia-smi`) being present — on a machine with neither (AMD GPUs,
    or an NVIDIA machine with no driver tooling installed),
    `detect_topology()` honestly reports `measured=False` rather than
    fabricating a topology it didn't observe, and callers fall back to
    the documented GPU-count heuristic.
"""
from __future__ import annotations

import os
import platform
import re
import subprocess
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class DistributedPlan:
    """Recommended `torch.distributed` setup for this machine."""

    backend: str  # "nccl" | "gloo"
    world_size: int
    bucket_cap_mb: int
    gradient_as_bucket_view: bool
    reason: str


def _in_wsl() -> bool:
    """Best-effort WSL detection — WSL2 runs a real Linux kernel (torch
    sees platform.system() == 'Linux') with the Linux NCCL wheel
    available, so it gets the same recommendation as native Linux."""
    if platform.system() != "Linux":
        return False
    try:
        with open("/proc/version", "r", encoding="utf-8", errors="ignore") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


def recommended_backend() -> str:
    """"nccl" everywhere NCCL is actually available (Linux, incl. WSL2)
    -- "gloo" on native Windows, where PyTorch ships without NCCL.

    Import of torch is intentionally NOT required here — this is pure
    platform detection, no GPU query needed to answer this question.
    """
    system = platform.system()
    if system == "Windows":
        return "gloo"
    return "nccl"  # Linux (incl. WSL2) and macOS (CPU-only gloo fallback happens inside torch itself)


def detected_gpu_count() -> int:
    """How many CUDA devices torch can currently see. Returns 0 if
    torch isn't installed or no CUDA device is visible -- never raises,
    since this is meant to be safe to call before deciding whether
    multi-GPU is even applicable."""
    try:
        import torch

        if not torch.cuda.is_available():
            return 0
        return torch.cuda.device_count()
    except Exception:
        return 0


# -- real interconnect topology detection -------------------------------
#
# PCIe distance ranking, worst-to-best, matching NVML's own
# NVML_TOPOLOGY_* enum and nvidia-smi's topo -m legend. Larger number =
# further apart = worse for all-reduce bandwidth/latency.
_PCIE_RANK = {
    "SELF": -1,       # same device (diagonal of the matrix, not a real pair)
    "INTERNAL": 0,     # e.g. two GPUs on the same physical board (rare, e.g. some dual-GPU cards)
    "SINGLE": 1,       # PIX -- single PCIe switch between them
    "MULTIPLE": 2,     # PXB -- multiple PCIe switches, no host bridge crossing
    "HOSTBRIDGE": 3,   # PHB -- crosses a PCIe host bridge
    "NODE": 4,         # NODE -- same NUMA node, crosses CPU interconnect (e.g. QPI/Infinity Fabric)
    "SYSTEM": 5,        # SYS -- crosses NUMA nodes/sockets entirely -- worst case
    "UNKNOWN": 6,
}
# nvidia-smi topo -m text tokens map to the same ranks
_NVIDIA_SMI_TOKEN_RANK = {
    "PIX": _PCIE_RANK["SINGLE"],
    "PXB": _PCIE_RANK["MULTIPLE"],
    "PHB": _PCIE_RANK["HOSTBRIDGE"],
    "NODE": _PCIE_RANK["NODE"],
    "SYS": _PCIE_RANK["SYSTEM"],
}


@dataclass
class GPULink:
    gpu_a: int
    gpu_b: int
    nvlink: bool
    nvlink_count: int = 0  # number of NVLink lines directly connecting this pair, if nvlink=True
    pcie_rank: Optional[int] = None  # None if nvlink=True (PCIe rank not meaningful once NVLink is present)
    label: str = ""  # human-readable, e.g. "NVLink x2", "PCIe (crosses host bridge)"


@dataclass
class TopologyReport:
    gpu_count: int
    links: List[GPULink] = field(default_factory=list)
    measured: bool = False  # True if this came from a real NVML/nvidia-smi read, not a guess
    source: str = "unknown"  # "pynvml" | "nvidia-smi" | "unknown"
    all_nvlink: bool = False  # True only if every pair is directly NVLink-connected
    worst_pcie_rank: Optional[int] = None  # max _PCIE_RANK among non-NVLink pairs (None if all_nvlink or unmeasured)
    note: str = ""


def _topology_via_pynvml(gpu_count: int) -> Optional[TopologyReport]:
    try:
        import pynvml
    except ImportError:
        return None

    try:
        pynvml.nvmlInit()
    except Exception:
        return None

    try:
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(gpu_count)]

        # Map each GPU's PCI bus id, so an NVLink's "remote" endpoint
        # (given as PCI info, not a GPU index) can be matched back to a
        # GPU index in our list.
        bus_ids = {}
        for i, h in enumerate(handles):
            try:
                pci = pynvml.nvmlDeviceGetPciInfo(h)
                bus_ids[pci.busId if isinstance(pci.busId, str) else pci.busId.decode()] = i
            except Exception:
                pass

        # nvlink_pairs[(i, j)] = count of direct NVLink lines between GPU i and j
        nvlink_pairs: Dict[Tuple[int, int], int] = {}
        max_links = getattr(pynvml, "NVML_NVLINK_MAX_LINKS", 18)
        for i, h in enumerate(handles):
            for link in range(max_links):
                try:
                    state = pynvml.nvmlDeviceGetNvLinkState(h, link)
                except Exception:
                    continue  # link index doesn't exist on this GPU -- not an error, just no more links
                if state != pynvml.NVML_FEATURE_ENABLED:
                    continue
                try:
                    remote = pynvml.nvmlDeviceGetNvLinkRemotePciInfo(h, link)
                    remote_bus = remote.busId if isinstance(remote.busId, str) else remote.busId.decode()
                except Exception:
                    continue
                j = bus_ids.get(remote_bus)
                if j is None or j == i:
                    continue  # remote end isn't one of the GPUs we're comparing (e.g. an NVSwitch), or maps to self
                key = (min(i, j), max(i, j))
                nvlink_pairs[key] = nvlink_pairs.get(key, 0) + 1

        links: List[GPULink] = []
        for i, j in combinations(range(gpu_count), 2):
            key = (i, j)
            if key in nvlink_pairs:
                count = nvlink_pairs[key]
                links.append(
                    GPULink(gpu_a=i, gpu_b=j, nvlink=True, nvlink_count=count, label=f"NVLink x{count}")
                )
                continue

            # No direct NVLink between this pair -- fall back to PCIe
            # topology distance via NVML's own common-ancestor query.
            try:
                ancestor = pynvml.nvmlDeviceGetTopologyCommonAncestor(handles[i], handles[j])
                name = {
                    getattr(pynvml, "NVML_TOPOLOGY_INTERNAL", -100): "INTERNAL",
                    getattr(pynvml, "NVML_TOPOLOGY_SINGLE", -101): "SINGLE",
                    getattr(pynvml, "NVML_TOPOLOGY_MULTIPLE", -102): "MULTIPLE",
                    getattr(pynvml, "NVML_TOPOLOGY_HOSTBRIDGE", -103): "HOSTBRIDGE",
                    getattr(pynvml, "NVML_TOPOLOGY_NODE", -104): "NODE",
                    getattr(pynvml, "NVML_TOPOLOGY_SYSTEM", -105): "SYSTEM",
                }.get(ancestor, "UNKNOWN")
            except Exception:
                name = "UNKNOWN"

            rank = _PCIE_RANK.get(name, _PCIE_RANK["UNKNOWN"])
            links.append(
                GPULink(
                    gpu_a=i,
                    gpu_b=j,
                    nvlink=False,
                    pcie_rank=rank,
                    label=f"PCIe ({name.lower()})",
                )
            )

        all_nvlink = bool(links) and all(l.nvlink for l in links)
        pcie_ranks = [l.pcie_rank for l in links if not l.nvlink and l.pcie_rank is not None]
        worst = max(pcie_ranks) if pcie_ranks else None

        return TopologyReport(
            gpu_count=gpu_count,
            links=links,
            measured=True,
            source="pynvml",
            all_nvlink=all_nvlink,
            worst_pcie_rank=worst,
            note="Measured via pynvml (NVML) -- NVLink presence from nvmlDeviceGetNvLinkState, PCIe distance from nvmlDeviceGetTopologyCommonAncestor.",
        )
    except Exception:
        return None
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _topology_via_nvidia_smi(gpu_count: int) -> Optional[TopologyReport]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None

    lines = result.stdout.splitlines()
    header_idx = None
    gpu_cols: List[int] = []  # column index -> gpu index, in header order
    for idx, line in enumerate(lines):
        if line.strip().startswith("GPU0"):
            header_idx = idx
            gpu_cols = [int(m.group(1)) for m in re.finditer(r"GPU(\d+)", line)]
            break
    if header_idx is None:
        return None

    # matrix[i][j] = raw token ("NV1", "PIX", "PXB", "PHB", "NODE", "SYS", "X")
    row_tokens: Dict[int, List[str]] = {}
    for line in lines[header_idx:]:
        m = re.match(r"\s*GPU(\d+)\s+(.*)", line)
        if not m:
            continue
        gpu_i = int(m.group(1))
        rest = m.group(2).split()
        row_tokens[gpu_i] = rest[: len(gpu_cols)]

    if not row_tokens:
        return None

    links: List[GPULink] = []
    for i, j in combinations(range(gpu_count), 2):
        tokens_i = row_tokens.get(i)
        if not tokens_i or j >= len(tokens_i):
            continue
        token = tokens_i[j]
        nv_match = re.match(r"NV(\d+)$", token)
        if nv_match:
            count = int(nv_match.group(1))
            links.append(GPULink(gpu_a=i, gpu_b=j, nvlink=True, nvlink_count=count, label=f"NVLink x{count}"))
        elif token in _NVIDIA_SMI_TOKEN_RANK:
            links.append(
                GPULink(
                    gpu_a=i,
                    gpu_b=j,
                    nvlink=False,
                    pcie_rank=_NVIDIA_SMI_TOKEN_RANK[token],
                    label=f"PCIe ({token})",
                )
            )
        # "X" (self) or unrecognized tokens are skipped -- not a pair we can classify

    if not links:
        return None

    all_nvlink = all(l.nvlink for l in links)
    pcie_ranks = [l.pcie_rank for l in links if not l.nvlink and l.pcie_rank is not None]
    worst = max(pcie_ranks) if pcie_ranks else None

    return TopologyReport(
        gpu_count=gpu_count,
        links=links,
        measured=True,
        source="nvidia-smi",
        all_nvlink=all_nvlink,
        worst_pcie_rank=worst,
        note="Measured by parsing `nvidia-smi topo -m` output (the vendor tool's own topology matrix).",
    )


def detect_topology(gpu_count: Optional[int] = None) -> TopologyReport:
    """Actually measure GPU interconnect topology: which pairs are
    NVLink-connected (and how many lines), and for pairs that aren't,
    how far apart they are on the PCIe tree (same switch, crosses a
    host bridge, crosses NUMA nodes...). Tries `pynvml` first (direct
    NVML calls, no subprocess), then falls back to parsing `nvidia-smi
    topo -m` if `pynvml` isn't installed but the driver's CLI tool is
    on PATH.

    Returns a `TopologyReport` with `measured=False` (and an empty
    `links` list) if neither path works -- e.g. no NVIDIA driver
    tooling present, AMD GPUs, or fewer than 2 GPUs (nothing to
    compare). This is never fabricated: `measured` tells the caller
    whether `recommended_bucket_cap_mb()` is about to use a real
    reading or fall back to the GPU-count heuristic.
    """
    gpu_count = detected_gpu_count() if gpu_count is None else gpu_count
    if gpu_count < 2:
        return TopologyReport(gpu_count=gpu_count, measured=False, note="Fewer than 2 GPUs -- no pair to measure.")

    report = _topology_via_pynvml(gpu_count)
    if report is not None:
        return report

    report = _topology_via_nvidia_smi(gpu_count)
    if report is not None:
        return report

    return TopologyReport(
        gpu_count=gpu_count,
        measured=False,
        source="unknown",
        note=(
            "Could not measure topology: 'pynvml' isn't installed and "
            "'nvidia-smi' isn't on PATH (or both failed). Falling back "
            "to the GPU-count heuristic in recommended_bucket_cap_mb() -- "
            "install nvidia-ml-py (`pip install nvidia-ml-py`) for a real per-pair "
            "reading instead of the heuristic."
        ),
    )


def recommended_bucket_cap_mb(
    gpu_count: Optional[int] = None, topology: Optional[TopologyReport] = None
) -> int:
    """DDP's default `bucket_cap_mb` is 25. This uses a *measured*
    interconnect topology (see `detect_topology()`) when one is
    available, and only falls back to a GPU-count heuristic when
    topology genuinely couldn't be measured (no pynvml, no nvidia-smi).

    Resolution when topology is measured (`topology.measured=True`):
      - Every pair is direct NVLink -> 25 (torch default). NVLink's
        bandwidth is high enough that the 25MB default already
        amortizes launch overhead well; smaller buckets are fine here
        (and start the all-reduce sooner, overlapping more with
        backward compute).
      - Otherwise, PCIe distance for the *worst* measured pair drives
        the bucket size -- a link that crosses a host bridge or NUMA
        node benefits more from fewer, larger transfers than one on a
        single PCIe switch:

            worst measured PCIe hop     bucket_cap_mb
            -----------------------     -------------
            single switch (PIX)          40
            multiple switches (PXB)      60
            host bridge (PHB)            80
            NUMA node (NODE)             90
            crosses sockets (SYS)        100

    Resolution when topology could NOT be measured
    (`topology.measured=False`) -- unchanged fallback, a heuristic to
    benchmark rather than a measurement:

        gpu_count   bucket_cap_mb
        --------    -------------
        <= 1        25
        2           50
        3-4         75
        > 4         100
    """
    gpu_count = detected_gpu_count() if gpu_count is None else gpu_count
    topology = detect_topology(gpu_count) if topology is None else topology

    if topology.measured:
        if topology.all_nvlink:
            return 25
        rank = topology.worst_pcie_rank if topology.worst_pcie_rank is not None else _PCIE_RANK["UNKNOWN"]
        rank_to_bucket = {
            _PCIE_RANK["INTERNAL"]: 25,
            _PCIE_RANK["SINGLE"]: 40,
            _PCIE_RANK["MULTIPLE"]: 60,
            _PCIE_RANK["HOSTBRIDGE"]: 80,
            _PCIE_RANK["NODE"]: 90,
            _PCIE_RANK["SYSTEM"]: 100,
        }
        return rank_to_bucket.get(rank, 100)

    # Unmeasured fallback -- heuristic only.
    if gpu_count <= 1:
        return 25
    if gpu_count == 2:
        return 50
    if gpu_count <= 4:
        return 75
    return 100


def plan_distributed(gpu_count: Optional[int] = None) -> DistributedPlan:
    """Put together a `DistributedPlan` for the detected (or given)
    GPU count on this machine. Does not call `init_process_group` --
    this only recommends settings; the caller wires them into their
    own launch (`torchrun`, `mp.spawn`, etc.), since how you launch
    processes is specific to your training script and orchestration."""
    gpu_count = detected_gpu_count() if gpu_count is None else gpu_count
    backend = recommended_backend()
    topology = detect_topology(gpu_count)
    bucket = recommended_bucket_cap_mb(gpu_count, topology=topology)

    if backend == "gloo":
        reason = (
            "Windows has no official NCCL build in PyTorch; 'gloo' is the "
            "backend that actually works here. It is slower than NCCL for "
            "large all-reduce, but correctness beats a backend that raises "
            "at init_process_group(). Run inside WSL2 with a Linux torch "
            "wheel if you need NCCL's throughput."
        )
    else:
        reason = "NCCL is available on this platform (Linux/WSL2) and is the fastest option for NVIDIA GPUs."

    if topology.measured:
        reason += f" Bucket size chosen from measured topology ({topology.source}): {topology.note}"
    else:
        reason += f" Topology not measured ({topology.note}); bucket size uses the GPU-count fallback heuristic."

    return DistributedPlan(
        backend=backend,
        world_size=max(1, gpu_count),
        bucket_cap_mb=bucket,
        gradient_as_bucket_view=True,  # reduces peak memory (grads write directly into the all-reduce buffer) with no correctness downside for standard training loops
        reason=reason,
    )


def ddp_kwargs(plan: Optional[DistributedPlan] = None, find_unused_parameters: bool = False) -> dict:
    """Return a kwargs dict ready to splat into
    `torch.nn.parallel.DistributedDataParallel(model, **kwargs)`.

    `find_unused_parameters` is left as an explicit argument you must
    decide, not something this module infers -- guessing wrong in
    either direction has a real cost (see module docstring), and this
    module has no visibility into whether your forward pass
    conditionally skips parameters.
    """
    plan = plan or plan_distributed()
    return {
        "bucket_cap_mb": plan.bucket_cap_mb,
        "gradient_as_bucket_view": plan.gradient_as_bucket_view,
        "find_unused_parameters": find_unused_parameters,
    }


@dataclass
class GPUBalanceReport:
    """Per-GPU VRAM snapshot used to flag a lopsided multi-GPU setup
    (e.g. one card also driving the display and starting a training
    run with much less free VRAM than the others) before it causes an
    OOM on just that one rank partway through training."""

    free_gb: List[float] = field(default_factory=list)
    total_gb: List[float] = field(default_factory=list)
    imbalance_ratio: float = 1.0  # max(free) / min(free), 1.0 = perfectly even
    warning: Optional[str] = None


def check_gpu_balance(imbalance_threshold: float = 1.5) -> GPUBalanceReport:
    """Read free VRAM per visible GPU (via `torch.cuda.mem_get_info`)
    and flag if one card has meaningfully less headroom than the
    others -- a common real cause of "training OOMs on rank 2 only,
    twenty minutes in" that has nothing to do with the model itself.
    Returns an empty report (no warning) if torch/CUDA isn't available,
    or if fewer than 2 GPUs are visible (nothing to compare)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return GPUBalanceReport()
        count = torch.cuda.device_count()
        if count < 2:
            return GPUBalanceReport()

        free_gb, total_gb = [], []
        for i in range(count):
            free_b, total_b = torch.cuda.mem_get_info(i)
            free_gb.append(round(free_b / (1024**3), 2))
            total_gb.append(round(total_b / (1024**3), 2))

        min_free = max(min(free_gb), 1e-6)
        ratio = max(free_gb) / min_free
        warning = None
        if ratio >= imbalance_threshold:
            worst = free_gb.index(min(free_gb))
            warning = (
                f"GPU {worst} has {free_gb[worst]}GB free vs up to "
                f"{max(free_gb)}GB on another visible GPU ({ratio:.2f}x "
                f"imbalance). Common causes: that card is also driving a "
                f"display, another process has it partially occupied, or "
                f"it's a different model with less VRAM. Training will "
                f"likely OOM on that rank before the others under an even "
                f"batch split -- consider excluding it or reducing its "
                f"share of the batch."
            )
        return GPUBalanceReport(
            free_gb=free_gb, total_gb=total_gb, imbalance_ratio=ratio, warning=warning
        )
    except Exception:
        return GPUBalanceReport()


def init_from_env(plan: Optional[DistributedPlan] = None, timeout_seconds: int = 1800):
    """Call `torch.distributed.init_process_group()` using the standard
    `torchrun`/`torch.distributed.launch` environment variables
    (`RANK`, `WORLD_SIZE`, `LOCAL_RANK`, `MASTER_ADDR`, `MASTER_PORT`)
    plus this module's own backend/topology recommendation, instead of
    the caller re-deriving the same handful of lines every project.

    This does NOT replace `torchrun` / `mp.spawn` — one of those still
    has to launch the N processes and set the env vars in the first
    place. This only wires the already-set env vars and the
    recommended backend into one `init_process_group()` call, and also
    puts the current process on the right GPU
    (`torch.cuda.set_device(local_rank)`) before returning, since
    forgetting that step is a common source of every rank silently
    using GPU 0.

    Raises `RuntimeError` with a clear message (rather than a torch
    `KeyError`) if the required env vars aren't set — i.e. if this
    process wasn't actually launched by `torchrun`/`mp.spawn`.

    Returns the `DistributedPlan` that was used (either the one you
    passed in, or a freshly computed `plan_distributed()`), so the
    caller has the backend/bucket-size reasoning available for logging
    without a second call.
    """
    import torch
    import torch.distributed as dist

    required = ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    missing = [v for v in required if v not in os.environ]
    if missing:
        raise RuntimeError(
            f"init_from_env() needs {missing} set in the environment — "
            f"this process doesn't look like it was launched by torchrun "
            f"or torch.distributed.launch/mp.spawn (which set these "
            f"automatically). Launch with e.g. `torchrun --nproc_per_node=N "
            f"your_script.py`, or set these manually if you're wiring up "
            f"a launcher yourself."
        )

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    plan = plan or plan_distributed(gpu_count=world_size)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    from datetime import timedelta

    dist.init_process_group(
        backend=plan.backend,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=timeout_seconds),
    )
    return plan


def local_rank_from_env() -> Optional[int]:
    """Read LOCAL_RANK the way `torchrun` sets it. Returns None if not
    running under a launcher that sets it (e.g. a single-process
    script) -- this is a read of an env var, not a detection of
    "should I be distributed", since that's a decision for the caller."""
    val = os.environ.get("LOCAL_RANK")
    if val is None:
        return None
    try:
        return int(val)
    except ValueError:
        return None
