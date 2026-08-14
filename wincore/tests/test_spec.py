from WinCore.spec import SystemSpec, GPUInfo, meets_minimum


def _fake_spec(**overrides):
    base = dict(
        os_name="Windows",
        logical_threads=12,
        physical_cores=6,
        ram_total_gb=32.0,
        ram_available_gb=20.0,
        gpus=[GPUInfo(name="NVIDIA GeForce RTX 3060", vendor="nvidia", vram_total_gb=12.0)],
    )
    base.update(overrides)
    return SystemSpec(**base)


def test_passes_when_above_all_minimums():
    result = meets_minimum(
        min_logical_threads=8,
        min_vram_gb=6,
        min_ram_gb=16,
        spec=_fake_spec(),
    )
    assert result.ok
    assert result.reasons == []


def test_fails_on_low_vram():
    result = meets_minimum(
        min_vram_gb=16,
        spec=_fake_spec(gpus=[GPUInfo(name="GTX 1060", vendor="nvidia", vram_total_gb=6.0)]),
    )
    assert not result.ok
    assert any("VRAM" in r for r in result.reasons)


def test_fails_when_no_gpu_detected():
    result = meets_minimum(min_vram_gb=6, spec=_fake_spec(gpus=[]))
    assert not result.ok
    assert any("no GPU detected" in r for r in result.reasons)


def test_fails_on_low_thread_count():
    result = meets_minimum(min_logical_threads=16, spec=_fake_spec(logical_threads=8))
    assert not result.ok
    assert any("logical threads" in r for r in result.reasons)


def test_unknown_vram_for_non_nvidia_is_reported_not_silently_passed():
    result = meets_minimum(
        min_vram_gb=6,
        spec=_fake_spec(gpus=[GPUInfo(name="AMD Radeon RX 6600", vendor="amd", vram_total_gb=None)]),
    )
    assert not result.ok
    assert any("VRAM size unknown" in r for r in result.reasons)
