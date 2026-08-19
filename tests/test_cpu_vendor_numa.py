"""
Tests for WinCore.cpu.cpu_vendor() / numa_node_count() /
numa_node_cpus() and the NUMA-aware branch of _select_pin_cpus().

Windows-only ctypes paths (numa_node_count/numa_node_cpus's real
mechanism) can't be exercised on this Linux sandbox, so those are
tested via monkeypatching platform.system() to "Windows" plus faking
the ctypes calls at the level _select_pin_cpus actually consumes
(numa_node_count/numa_node_cpus themselves), same convention as the
existing P-core tests monkeypatch _detect_windows_performance_cores.
"""
import platform

import pytest

from WinCore import cpu as cpu_module
from WinCore.cpu import cpu_vendor, numa_node_count, numa_node_cpus, _select_pin_cpus


# -- cpu_vendor -------------------------------------------------------------


def test_cpu_vendor_detects_amd(monkeypatch):
    monkeypatch.setattr(platform, "processor", lambda: "AMD64 Family 25 Model 33 Stepping 2, AuthenticAMD")
    assert cpu_vendor() == "amd"


def test_cpu_vendor_detects_intel(monkeypatch):
    monkeypatch.setattr(platform, "processor", lambda: "Intel64 Family 6 Model 154 Stepping 3, GenuineIntel")
    assert cpu_vendor() == "intel"


def test_cpu_vendor_unknown_for_unrecognized_string(monkeypatch):
    monkeypatch.setattr(platform, "processor", lambda: "some_weird_arm_string")
    assert cpu_vendor() == "unknown"


def test_cpu_vendor_unknown_for_empty_string(monkeypatch):
    monkeypatch.setattr(platform, "processor", lambda: "")
    assert cpu_vendor() == "unknown"


def test_cpu_vendor_never_raises_on_processor_exception(monkeypatch):
    def raise_err():
        raise OSError("simulated failure")

    monkeypatch.setattr(platform, "processor", raise_err)
    assert cpu_vendor() == "unknown"


# -- numa_node_count / numa_node_cpus (non-Windows path) --------------------


def test_numa_node_count_non_windows_returns_1(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    assert numa_node_count() == 1


def test_numa_node_cpus_non_windows_returns_none(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    assert numa_node_cpus(0) is None


# -- _select_pin_cpus NUMA-aware branch --------------------------------------
# These test the *consuming* logic in _select_pin_cpus by faking
# numa_node_count/numa_node_cpus and _detect_windows_performance_cores
# directly (same pattern test_cpu.py already uses for P-core tests),
# rather than faking the underlying ctypes calls.


def test_select_pin_cpus_restricts_to_numa_node_when_multi_node(monkeypatch):
    monkeypatch.setattr(cpu_module, "numa_node_count", lambda: 2)
    monkeypatch.setattr(cpu_module, "numa_node_cpus", lambda node: [0, 1, 2, 3] if node == 0 else [4, 5, 6, 7])
    monkeypatch.setattr(cpu_module, "_detect_windows_performance_cores", lambda: None)  # no hybrid P-cores

    cpus = _select_pin_cpus(recommended=2, numa_aware=True)
    assert cpus == [0, 1]  # from node 0's pool, not scattered across both nodes


def test_select_pin_cpus_single_node_uses_plain_range(monkeypatch):
    monkeypatch.setattr(cpu_module, "numa_node_count", lambda: 1)
    monkeypatch.setattr(cpu_module, "_detect_windows_performance_cores", lambda: None)

    cpus = _select_pin_cpus(recommended=4, numa_aware=True)
    assert cpus == [0, 1, 2, 3]


def test_select_pin_cpus_numa_aware_false_ignores_numa(monkeypatch):
    monkeypatch.setattr(cpu_module, "numa_node_count", lambda: 2)
    monkeypatch.setattr(cpu_module, "numa_node_cpus", lambda node: [0, 1, 2, 3])
    monkeypatch.setattr(cpu_module, "_detect_windows_performance_cores", lambda: None)

    cpus = _select_pin_cpus(recommended=6, numa_aware=False)
    assert cpus == [0, 1, 2, 3, 4, 5]  # plain range, NUMA restriction skipped entirely


def test_select_pin_cpus_numa_and_p_core_combined(monkeypatch):
    # P-cores span both NUMA nodes; NUMA restriction to node 0 should
    # apply BEFORE truncating to `recommended`.
    monkeypatch.setattr(cpu_module, "numa_node_count", lambda: 2)
    monkeypatch.setattr(cpu_module, "numa_node_cpus", lambda node: [0, 1, 2, 3] if node == 0 else [4, 5, 6, 7])
    monkeypatch.setattr(cpu_module, "_detect_windows_performance_cores", lambda: [1, 3, 5, 7])  # P-cores, mixed nodes

    cpus = _select_pin_cpus(recommended=2, numa_aware=True)
    assert cpus == [1, 3]  # only the node-0 P-cores (1, 3), not 5 or 7


def test_select_pin_cpus_numa_restriction_falls_back_if_empty_intersection(monkeypatch):
    # If restricting P-cores to node 0 would leave NOTHING usable,
    # fall back to the full P-core pool rather than pinning to nothing.
    monkeypatch.setattr(cpu_module, "numa_node_count", lambda: 2)
    monkeypatch.setattr(cpu_module, "numa_node_cpus", lambda node: [0, 1, 2, 3] if node == 0 else [4, 5, 6, 7])
    monkeypatch.setattr(cpu_module, "_detect_windows_performance_cores", lambda: [4, 5, 6, 7])  # all node 1

    cpus = _select_pin_cpus(recommended=2, numa_aware=True)
    assert cpus == [4, 5]  # fell back to the full (unrestricted) P-core pool


def test_select_pin_cpus_numa_node_cpus_unavailable_falls_back(monkeypatch):
    monkeypatch.setattr(cpu_module, "numa_node_count", lambda: 2)
    monkeypatch.setattr(cpu_module, "numa_node_cpus", lambda node: None)  # couldn't determine
    monkeypatch.setattr(cpu_module, "_detect_windows_performance_cores", lambda: None)

    cpus = _select_pin_cpus(recommended=4, numa_aware=True)
    assert cpus == [0, 1, 2, 3]  # plain range fallback, no crash


# -- ThreadPlan.vendor -------------------------------------------------------


def test_recommended_threads_populates_vendor(monkeypatch):
    monkeypatch.setattr(platform, "processor", lambda: "AMD64 Family 25, AuthenticAMD")
    from WinCore.cpu import recommended_threads

    plan = recommended_threads(total=8)
    assert plan.vendor == "amd"
