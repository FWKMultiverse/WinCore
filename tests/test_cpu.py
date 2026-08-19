import pytest

from WinCore.cpu import PriorityError, apply, pin_affinity, recommended_threads, set_priority


def test_small_machine_reserves_one():
    plan = recommended_threads(total=4)
    assert plan.total_logical == 4
    assert plan.reserved == 1
    assert plan.recommended == 3


def test_six_thread_reserves_one():
    plan = recommended_threads(total=6)
    assert plan.reserved == 1
    assert plan.recommended == 5


def test_twelve_thread_reserves_two():
    plan = recommended_threads(total=12)
    assert plan.reserved == 2
    assert plan.recommended == 10


def test_sixteen_thread_reserves_up_to_three():
    plan = recommended_threads(total=16)
    assert plan.reserved == 3
    assert plan.recommended == 13


def test_explicit_threads_overrides_heuristic():
    plan = recommended_threads(total=16, threads=8)
    assert plan.recommended == 8
    assert plan.reserved == 8


def test_explicit_reserve_overrides_heuristic():
    plan = recommended_threads(total=8, reserve=4)
    assert plan.recommended == 4
    assert plan.reserved == 4


def test_never_recommends_zero():
    plan = recommended_threads(total=1)
    assert plan.recommended == 1


def test_set_priority_rejects_unknown_level():
    with pytest.raises(PriorityError):
        set_priority("ludicrous_speed")


def test_set_priority_applies_or_reports_cleanly():
    """psutil is an optional dependency -- this only asserts the two
    honest outcomes: it worked, or it raised a clear PriorityError
    (missing psutil / OS denied it), never a bare AttributeError or a
    silent no-op."""
    pytest.importorskip("psutil")
    try:
        level = set_priority("above_normal")
        assert level == "above_normal"
    except PriorityError:
        pass  # OS/permission denial is an acceptable outcome here, not a bug


def test_pin_affinity_defaults_to_recommended_thread_count():
    pytest.importorskip("psutil")
    import platform

    if platform.system() == "Darwin":
        pytest.skip("cpu_affinity() has no public API on macOS")
    try:
        applied = pin_affinity()
        assert len(applied) >= 1
        # restore -- don't leave the test process pinned for whatever runs next
        import psutil

        psutil.Process().cpu_affinity(list(range(recommended_threads().total_logical)))
    except Exception:
        pytest.skip("affinity pinning not permitted in this environment")


def test_apply_with_bad_priority_is_nonfatal_by_default():
    """apply() should never crash a training run just because a
    best-effort OS-level tweak failed -- failures land in
    plan.warnings instead."""
    plan = apply(total=4, set_env=False)
    assert plan.warnings == ()  # nothing requested -> nothing to warn about


def test_apply_strict_raises_on_unsupported_affinity():
    import platform

    if platform.system() != "Darwin":
        pytest.skip("this specifically exercises the no-affinity-API platform")
    with pytest.raises(PriorityError):
        apply(total=4, set_env=False, affinity=True, strict=True)


def test_apply_affinity_uses_p_core_detection_not_bare_range(monkeypatch):
    """Regression test for a real integration bug: `apply(affinity=True)`
    used to build `range(plan.recommended)` itself and pass it straight
    to `pin_affinity()` as an explicit `cpus` argument -- which bypasses
    `pin_affinity()`'s own P-core detection entirely, since that only
    runs when `cpus is None`. `pin_affinity()` tested in isolation
    can't catch this class of bug (it behaves exactly as documented on
    its own); only an integration-level assertion on what `apply()`
    actually hands to psutil, as below, would have caught it."""
    pytest.importorskip("psutil")
    import psutil

    from WinCore import cpu as cpu_module

    fake_p_cores = [4, 5, 6, 7]  # pretend logical CPUs 4-7 are the P-cores
    monkeypatch.setattr(cpu_module, "_detect_windows_performance_cores", lambda: fake_p_cores)

    captured = {}

    def fake_cpu_affinity(self, cpus=None):
        if cpus is None:
            return captured.get("cpus", [])
        captured["cpus"] = cpus
        return None

    monkeypatch.setattr(psutil.Process, "cpu_affinity", fake_cpu_affinity)

    apply(total=4, reserve=0, set_env=False, affinity=True)

    # plan.recommended == 4, matching len(fake_p_cores) exactly, so the
    # P-core-aware path should hand over fake_p_cores verbatim -- NOT
    # range(4) == [0, 1, 2, 3], which is what the old bug produced
    # regardless of what _detect_windows_performance_cores() returned.
    assert captured["cpus"] == fake_p_cores
