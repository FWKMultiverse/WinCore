"""
Tests for the MSVC auto-detection logic in WinCore.kernels.build
(_autodetect_and_setup_msvc and its helpers) -- the mechanism that lets
building the CUDA kernel work from a plain terminal or an IDE's
integrated terminal (VSCode, Cursor, PyCharm, ...) without needing to
specifically open "x64 Native Tools Command Prompt for VS" first.

Everything here is mocked (shutil.which, os.path.isfile,
subprocess.run) -- there is no real Visual Studio install, and no
Windows machine, in the environment these were first run in. What's
verified is the DECISION LOGIC: given a particular filesystem/
subprocess-output shape, does it correctly locate vswhere -> vcvars64
-> apply the right env vars -- not whether a real VS install on a real
machine actually behaves this way (that still needs testing on an
actual Windows+VS machine, same caveat as the rest of build.py).
"""
import os

import pytest

import WinCore.kernels.build as build


def test_returns_true_immediately_if_cl_already_on_path(monkeypatch):
    monkeypatch.setattr(build.shutil, "which", lambda name: "/fake/cl.exe" if name == "cl" else None)
    assert build._autodetect_and_setup_msvc() is True


def test_returns_false_on_non_windows_without_attempting_detection(monkeypatch):
    monkeypatch.setattr(build.shutil, "which", lambda name: None)
    monkeypatch.setattr(build.os, "name", "posix")
    assert build._autodetect_and_setup_msvc() is False


def test_returns_false_when_vswhere_cannot_be_located(monkeypatch):
    monkeypatch.setattr(build.shutil, "which", lambda name: None)
    monkeypatch.setattr(build.os, "name", "nt")
    monkeypatch.setattr(build.os.path, "isfile", lambda p: False)
    assert build._find_vswhere() is None
    assert build._autodetect_and_setup_msvc() is False


def test_returns_false_when_vcvars64_bat_missing_under_found_vs_install(monkeypatch):
    monkeypatch.setattr(build.os, "name", "nt")
    monkeypatch.setattr(build.shutil, "which", lambda name: None)
    program_files_x86 = "C:/Program Files (x86)"
    monkeypatch.setenv("ProgramFiles(x86)", program_files_x86)
    # built via os.path.join, not hardcoded -- see the comment in
    # test_full_happy_path... for why a literal string doesn't match
    # what _find_vswhere() actually constructs on real Windows
    vswhere_path = os.path.join(program_files_x86, "Microsoft Visual Studio", "Installer", "vswhere.exe")

    monkeypatch.setattr(build.os.path, "isfile", lambda p: p == vswhere_path)

    def fake_run(cmd, **kwargs):
        return type("P", (), {"stdout": "C:/VS\n", "returncode": 0})()

    monkeypatch.setattr(build.subprocess, "run", fake_run)
    # vswhere IS found (isfile matches it), but vcvars64.bat is not
    # (isfile only matches vswhere_path) -- confirms _find_vswhere
    # actually succeeds here, so the eventual False comes from the
    # vcvars64.bat check specifically, not a vswhere lookup failure
    assert build._find_vswhere() == vswhere_path
    assert build._find_vcvars64(vswhere_path) is None
    assert build._autodetect_and_setup_msvc() is False


def test_find_vcvars64_query_includes_prerelease_flag(monkeypatch):
    # Regression guard for the real VS2026-preview-channel finding:
    # without "-prerelease", vswhere only returns stable-channel VS
    # installs, so a preview/insider-only install (which VS2026 was at
    # the time) is silently invisible to detection.
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return type("P", (), {"stdout": "", "returncode": 0})()

    monkeypatch.setattr(build.subprocess, "run", fake_run)
    build._find_vcvars64("C:/fake/vswhere.exe")

    assert "-prerelease" in captured["cmd"]


def test_full_happy_path_applies_path_include_lib_and_finds_cl(monkeypatch):
    monkeypatch.setattr(build.os, "name", "nt")
    program_files_x86 = "C:/Program Files (x86)"
    monkeypatch.setenv("ProgramFiles(x86)", program_files_x86)

    # Built via os.path.join itself, not a hardcoded literal string --
    # os.path.join does NOT normalize separators already present in its
    # first argument, it only appends subsequent parts using the
    # platform's own separator. So on real Windows,
    # os.path.join("C:/Program Files (x86)", "Microsoft Visual Studio", ...)
    # produces a MIXED "/" and "\" path -- a literal all-forward-slash
    # string here would never match what _find_vswhere() actually
    # constructs internally with the same os.path.join call. Building
    # the expected value the same way the code under test builds it is
    # what makes this test correct on POSIX (this sandbox) AND on real
    # Windows (where this was actually caught failing) at the same time.
    vswhere_path = os.path.join(program_files_x86, "Microsoft Visual Studio", "Installer", "vswhere.exe")
    vswhere_install_output = "C:/VS"
    vcvars_path = os.path.join(vswhere_install_output, "VC", "Auxiliary", "Build", "vcvars64.bat")

    monkeypatch.setattr(build.os.path, "isfile", lambda p: p in (vswhere_path, vcvars_path))

    def fake_run(cmd, **kwargs):
        if isinstance(cmd, list) and cmd and cmd[0] == vswhere_path:
            return type("P", (), {"stdout": vswhere_install_output + "\n", "returncode": 0})()
        if isinstance(cmd, str) and "vcvars64.bat" in cmd:
            stdout = (
                "PATH=C:\\VS\\bin;C:\\other\n"
                "INCLUDE=C:\\VS\\include\n"
                "LIB=C:\\VS\\lib\n"
                "SOME_INTERNAL_VS_VAR=ignored\n"
            )
            return type("P", (), {"stdout": stdout, "returncode": 0})()
        raise AssertionError(f"unexpected subprocess.run call: {cmd!r}")

    monkeypatch.setattr(build.subprocess, "run", fake_run)

    call_count = {"n": 0}

    def fake_which(name):
        call_count["n"] += 1
        # first call (the initial "already on PATH?" check) -> not found;
        # subsequent calls (after vcvars applied) -> found
        if name == "cl":
            return "C:/VS/bin/cl.exe" if call_count["n"] > 1 else None
        return None

    monkeypatch.setattr(build.shutil, "which", fake_which)

    result = build._autodetect_and_setup_msvc()

    assert result is True
    assert os.environ["PATH"] == "C:\\VS\\bin;C:\\other"
    assert os.environ["INCLUDE"] == "C:\\VS\\include"
    assert os.environ["LIB"] == "C:\\VS\\lib"


def test_apply_vcvars64_env_returns_false_on_nonzero_returncode(monkeypatch):
    def fake_run(cmd, **kwargs):
        return type("P", (), {"stdout": "", "returncode": 1})()

    monkeypatch.setattr(build.subprocess, "run", fake_run)
    assert build._apply_vcvars64_env("C:/fake/vcvars64.bat") is False


def test_apply_vcvars64_env_returns_false_on_subprocess_exception(monkeypatch):
    def fake_run(cmd, **kwargs):
        raise OSError("simulated failure launching cmd.exe")

    monkeypatch.setattr(build.subprocess, "run", fake_run)
    assert build._apply_vcvars64_env("C:/fake/vcvars64.bat") is False


def test_check_cl_raises_actionable_error_when_autodetect_fails(monkeypatch):
    monkeypatch.setattr(build, "_autodetect_and_setup_msvc", lambda: False)
    monkeypatch.setattr(build.os, "name", "nt")
    with pytest.raises(RuntimeError, match="x64 Native Tools Command Prompt"):
        build._check_cl()


def test_check_cl_passes_silently_when_autodetect_succeeds(monkeypatch):
    monkeypatch.setattr(build, "_autodetect_and_setup_msvc", lambda: True)
    monkeypatch.setattr(build.os, "name", "nt")
    build._check_cl()  # should not raise


def test_check_cl_is_a_noop_on_non_windows(monkeypatch):
    monkeypatch.setattr(build.os, "name", "posix")
    build._check_cl()  # should not raise, should not even check cl
