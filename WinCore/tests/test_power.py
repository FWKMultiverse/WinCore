import types

import pytest

import WinCore.power as power
from WinCore.power import PowerError, TdrReport, check_tdr_risk, prevent_sleep


def test_prevent_sleep_is_a_real_noop_on_non_windows(monkeypatch):
    monkeypatch.setattr(power.platform, "system", lambda: "Linux")

    with prevent_sleep() as p:
        assert p.start() is False  # already started by __enter__; calling again is still a clean no-op
    # stop() on non-Windows must also be a silent no-op, not an AttributeError
    prevent_sleep().stop()


def test_prevent_sleep_calls_real_win32_api_with_expected_flags(monkeypatch):
    """Stubs ctypes.windll (this sandbox isn't Windows) to verify the
    exact flag combination sent to SetThreadExecutionState, and that
    stop() restores ES_CONTINUOUS alone (releasing the request)."""
    import ctypes

    monkeypatch.setattr(power.platform, "system", lambda: "Windows")

    calls = []

    class FakeKernel32:
        def SetThreadExecutionState(self, flags):
            calls.append(flags)
            return 0x80000000  # nonzero (the previous state) = success

    fake_windll = types.SimpleNamespace(kernel32=FakeKernel32())
    monkeypatch.setattr(ctypes, "windll", fake_windll, raising=False)

    with prevent_sleep(keep_display_on=False):
        pass

    assert calls[0] == (0x80000000 | 0x00000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    assert calls[1] == 0x80000000  # stop() restores ES_CONTINUOUS alone


def test_prevent_sleep_keep_display_on_sets_display_flag(monkeypatch):
    import ctypes

    monkeypatch.setattr(power.platform, "system", lambda: "Windows")

    calls = []

    class FakeKernel32:
        def SetThreadExecutionState(self, flags):
            calls.append(flags)
            return 0x80000000

    fake_windll = types.SimpleNamespace(kernel32=FakeKernel32())
    monkeypatch.setattr(ctypes, "windll", fake_windll, raising=False)

    with prevent_sleep(keep_display_on=True):
        pass

    assert calls[0] == (0x80000000 | 0x00000001 | 0x00000002)


def test_prevent_sleep_raises_on_win32_failure(monkeypatch):
    import ctypes

    monkeypatch.setattr(power.platform, "system", lambda: "Windows")

    class FailingKernel32:
        def SetThreadExecutionState(self, flags):
            return 0  # zero = failure, matches Win32 BOOL convention

    fake_windll = types.SimpleNamespace(kernel32=FailingKernel32())
    monkeypatch.setattr(ctypes, "windll", fake_windll, raising=False)
    monkeypatch.setattr(ctypes, "get_last_error", lambda: 5, raising=False)

    with pytest.raises(PowerError):
        prevent_sleep().start()


def test_check_tdr_risk_reports_not_applicable_on_non_windows(monkeypatch):
    monkeypatch.setattr(power.platform, "system", lambda: "Linux")

    report = check_tdr_risk()
    assert report.platform_is_windows is False
    assert report.tdr_delay_seconds is None
    assert report.at_default_risk_level is False


def test_check_tdr_risk_defaults_to_2s_when_registry_value_missing(monkeypatch):
    """A missing/unreadable TdrDelay means 'never configured', i.e.
    still on Windows' 2s default -- not 'no timeout'."""
    monkeypatch.setattr(power.platform, "system", lambda: "Windows")

    class FakeWinreg:
        HKEY_LOCAL_MACHINE = object()

        def OpenKey(self, *a, **k):
            raise FileNotFoundError("registry value not set")

    monkeypatch.setitem(__import__("sys").modules, "winreg", FakeWinreg())

    report = check_tdr_risk()
    assert report.tdr_delay_seconds == 2
    assert report.at_default_risk_level is True
    assert "TdrDelay" in report.message


def test_check_tdr_risk_reads_configured_value_above_default(monkeypatch):
    monkeypatch.setattr(power.platform, "system", lambda: "Windows")

    class FakeKey:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class FakeWinreg:
        HKEY_LOCAL_MACHINE = object()

        def OpenKey(self, *a, **k):
            return FakeKey()

        def QueryValueEx(self, key, name):
            assert name == "TdrDelay"
            return (10, 4)  # (value, reg_type) -- matches winreg.QueryValueEx's return shape

    monkeypatch.setitem(__import__("sys").modules, "winreg", FakeWinreg())

    report = check_tdr_risk()
    assert report.tdr_delay_seconds == 10
    assert report.at_default_risk_level is False


def test_tdr_report_is_a_plain_dataclass():
    # cheap structural check that the public shape hasn't drifted
    report = TdrReport(
        platform_is_windows=True, tdr_delay_seconds=2,
        at_default_risk_level=True, message="x",
    )
    assert report.tdr_delay_seconds == 2
