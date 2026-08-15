import time

from WinCore.diagnostics import TrainingMonitor


def test_detects_nan_loss_immediately():
    m = TrainingMonitor()
    issue = m.record_loss(step=5, loss_value=float("nan"))
    assert issue is not None
    assert issue.code == "loss_nan_or_inf"
    assert issue.severity == "critical"


def test_detects_inf_loss_immediately():
    m = TrainingMonitor()
    issue = m.record_loss(step=5, loss_value=float("inf"))
    assert issue is not None
    assert issue.code == "loss_nan_or_inf"


def test_normal_decreasing_loss_raises_no_issue():
    m = TrainingMonitor(loss_plateau_window=10)
    issue = None
    for step, loss in enumerate([10 - i * 0.5 for i in range(10)]):
        issue = m.record_loss(step, loss)
    assert issue is None


def test_detects_plateau():
    m = TrainingMonitor(loss_plateau_window=10, loss_plateau_min_relative_improvement=0.01)
    issue = None
    for step in range(10):
        issue = m.record_loss(step, 1.0000001)  # essentially flat
    assert issue is not None
    assert issue.code == "loss_plateau"


def test_on_issue_callback_fires():
    seen = []
    m = TrainingMonitor(on_issue=lambda i: seen.append(i))
    m.record_loss(0, float("nan"))
    assert len(seen) == 1
    assert seen[0].code == "loss_nan_or_inf"


def test_bottleneck_report_flags_data_bound_loop():
    m = TrainingMonitor()
    for _ in range(3):
        with m.data_timer():
            time.sleep(0.02)
        with m.compute_timer():
            time.sleep(0.01)
    report = m.bottleneck_report()
    assert report["data_fraction"] > 0.4
    assert any(i.code == "dataloader_bottleneck" for i in m.issues)


def test_bottleneck_report_does_not_flag_compute_bound_loop():
    m = TrainingMonitor()
    for _ in range(3):
        with m.data_timer():
            time.sleep(0.005)
        with m.compute_timer():
            time.sleep(0.03)
    report = m.bottleneck_report()
    assert report["data_fraction"] < 0.4
    assert not any(i.code == "dataloader_bottleneck" for i in m.issues)


def test_gpu_timer_without_cuda_is_inert():
    """No GPU in this sandbox -- this only checks that gpu_timer()
    degrades to a harmless wall-clock-only no-op when CUDA isn't
    available, not the actual torch.cuda.Event measurement path (which
    needs a real CUDA device and has not been run here)."""
    import pytest

    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("this test targets the no-CUDA fallback path specifically")

    m = TrainingMonitor()
    with m.gpu_timer():
        time.sleep(0.01)
    report = m.bottleneck_report()
    # No CUDA -> no event samples recorded -> these keys are simply absent,
    # not present-but-wrong.
    assert "gpu_idle_fraction" not in report


def test_summary_accumulates_all_issues():
    m = TrainingMonitor()
    m.record_loss(0, float("nan"))
    m.record_loss(1, float("inf"))
    assert len(m.summary()) == 2


def test_record_signal_annotates_nearby_issue():
    m = TrainingMonitor(signal_correlation_window=2)
    m.record_signal(step=4, name="gpu_temp_c", value=87)
    issue = m.record_loss(step=5, loss_value=float("nan"))
    assert issue is not None
    assert "nearby_signals" in issue.data
    assert any(s["name"] == "gpu_temp_c" for s in issue.data["nearby_signals"])
    assert "gpu_temp_c" in issue.message


def test_record_signal_outside_window_not_attached():
    m = TrainingMonitor(signal_correlation_window=1)
    m.record_signal(step=0, name="gpu_temp_c", value=87)
    issue = m.record_loss(step=10, loss_value=float("nan"))
    assert issue is not None
    assert "nearby_signals" not in issue.data


def test_no_signals_means_no_annotation():
    m = TrainingMonitor()
    issue = m.record_loss(step=1, loss_value=float("nan"))
    assert issue is not None
    assert "nearby_signals" not in issue.data
