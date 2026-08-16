from WinCore.multigpu import (
    recommended_bucket_cap_mb,
    plan_distributed,
    ddp_kwargs,
    local_rank_from_env,
    detect_topology,
    TopologyReport,
    GPULink,
    _PCIE_RANK,
    _topology_via_nvidia_smi,
)


def test_bucket_cap_falls_back_to_heuristic_when_unmeasured():
    unmeasured = TopologyReport(gpu_count=2, measured=False)
    assert recommended_bucket_cap_mb(gpu_count=1, topology=unmeasured) == 25
    assert recommended_bucket_cap_mb(gpu_count=2, topology=unmeasured) == 50
    assert recommended_bucket_cap_mb(gpu_count=4, topology=unmeasured) == 75
    assert recommended_bucket_cap_mb(gpu_count=8, topology=unmeasured) == 100


def test_bucket_cap_uses_measured_all_nvlink_topology():
    nvlink = TopologyReport(
        gpu_count=2,
        links=[GPULink(gpu_a=0, gpu_b=1, nvlink=True, nvlink_count=4)],
        measured=True,
        all_nvlink=True,
    )
    # measured NVLink beats the GPU-count heuristic's 50 -- real bandwidth
    # doesn't need larger buckets to amortize overhead.
    assert recommended_bucket_cap_mb(gpu_count=2, topology=nvlink) == 25


def test_bucket_cap_uses_measured_pcie_worst_case():
    sys_bound = TopologyReport(
        gpu_count=2,
        links=[GPULink(gpu_a=0, gpu_b=1, nvlink=False, pcie_rank=_PCIE_RANK["SYSTEM"])],
        measured=True,
        all_nvlink=False,
        worst_pcie_rank=_PCIE_RANK["SYSTEM"],
    )
    assert recommended_bucket_cap_mb(gpu_count=2, topology=sys_bound) == 100


def test_bucket_cap_uses_measured_pcie_best_case_lower_than_heuristic():
    single_switch = TopologyReport(
        gpu_count=2,
        links=[GPULink(gpu_a=0, gpu_b=1, nvlink=False, pcie_rank=_PCIE_RANK["SINGLE"])],
        measured=True,
        all_nvlink=False,
        worst_pcie_rank=_PCIE_RANK["SINGLE"],
    )
    # measured single-switch PCIe is a better case than the blind 2-GPU
    # heuristic (50) assumes -- should come in lower.
    assert recommended_bucket_cap_mb(gpu_count=2, topology=single_switch) == 40


def test_detect_topology_with_fewer_than_two_gpus_is_honestly_unmeasured():
    report = detect_topology(gpu_count=1)
    assert report.measured is False
    assert report.links == []


def test_nvidia_smi_topology_parser_reads_mixed_nvlink_and_pcie(monkeypatch):
    sample_output = (
        "\tGPU0\tGPU1\tGPU2\tGPU3\tCPU Affinity\tNUMA Affinity\n"
        "GPU0\t X \tNV1\tPXB\tPXB\t0-19\t0\n"
        "GPU1\tNV1\t X \tPXB\tPXB\t0-19\t0\n"
        "GPU2\tPXB\tPXB\t X \tNV2\t0-19\t0\n"
        "GPU3\tPXB\tPXB\tNV2\t X \t0-19\t0\n"
    )

    class FakeResult:
        returncode = 0
        stdout = sample_output
        stderr = ""

    import subprocess

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: FakeResult())

    report = _topology_via_nvidia_smi(4)
    assert report is not None
    assert report.measured is True
    assert report.source == "nvidia-smi"

    link_map = {(l.gpu_a, l.gpu_b): l for l in report.links}
    assert link_map[(0, 1)].nvlink is True and link_map[(0, 1)].nvlink_count == 1
    assert link_map[(2, 3)].nvlink is True and link_map[(2, 3)].nvlink_count == 2
    assert link_map[(0, 2)].nvlink is False
    assert link_map[(0, 2)].pcie_rank == _PCIE_RANK["MULTIPLE"]
    assert report.all_nvlink is False
    assert report.worst_pcie_rank == _PCIE_RANK["MULTIPLE"]


def test_nvidia_smi_topology_parser_returns_none_when_tool_missing(monkeypatch):
    import subprocess

    def _raise(*a, **k):
        raise FileNotFoundError("no nvidia-smi")

    monkeypatch.setattr(subprocess, "run", _raise)
    assert _topology_via_nvidia_smi(2) is None


def test_plan_distributed_single_gpu_still_returns_a_plan():
    plan = plan_distributed(gpu_count=1)
    assert plan.world_size == 1
    assert plan.backend in ("nccl", "gloo")
    assert plan.bucket_cap_mb == 25


def test_plan_distributed_never_reports_zero_world_size():
    plan = plan_distributed(gpu_count=0)
    assert plan.world_size == 1


def test_ddp_kwargs_respects_explicit_find_unused_parameters():
    plan = plan_distributed(gpu_count=3)
    kwargs = ddp_kwargs(plan, find_unused_parameters=True)
    assert kwargs["find_unused_parameters"] is True
    assert kwargs["gradient_as_bucket_view"] is True


def test_local_rank_from_env_reads_torchrun_var(monkeypatch):
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    assert local_rank_from_env() is None
    monkeypatch.setenv("LOCAL_RANK", "2")
    assert local_rank_from_env() == 2


def test_local_rank_from_env_ignores_garbage(monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "not-a-number")
    assert local_rank_from_env() is None
