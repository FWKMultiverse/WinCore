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
    monkeypatch.setattr(build.shutil, "which", lambda name: r"C:\fake\cl.exe" if name == "cl" else None)
    build._check_cl()  # should not raise


def test_check_cl_returns_the_resolved_cl_path(monkeypatch):
    """_check_cl()'s own docstring says build() feeds this straight to
    nvcc via -ccbin=<path> -- this asserts the return value itself is
    actually the resolved path, which is the half of that contract
    _check_cl() is responsible for (see test_build_uses_ccbin_flag
    below for the other half: that build() actually uses it)."""
    monkeypatch.setattr(build, "_autodetect_and_setup_msvc", lambda: True)
    monkeypatch.setattr(build.os, "name", "nt")
    monkeypatch.setattr(build.shutil, "which", lambda name: r"C:\fake\cl.exe" if name == "cl" else None)
    assert build._check_cl() == r"C:\fake\cl.exe"


def test_check_cl_is_a_noop_on_non_windows(monkeypatch):
    monkeypatch.setattr(build.os, "name", "posix")
    build._check_cl()  # should not raise, should not even check cl


# -- build() actually using _check_cl()'s returned path -----------------
#
# Regression coverage for the real bug this fixed: _check_cl() has
# always RESOLVED cl.exe's path (its own docstring said as much), but
# build() used to discard that return value and never passed it to
# nvcc at all. Harmless on most CUDA Toolkit + MSVC combinations
# (nvcc's own auto-detection finds cl.exe fine), but a real, confirmed
# failure on a VS2026 ("18.x"/v145 MSVC toolset) + CUDA 13.x machine,
# where nvcc's internal auto-detection didn't recognize that toolset
# version and failed to locate cl.exe even though it was genuinely on
# PATH -- surfacing as an unhelpful `CreateProcess failed: The system
# cannot find the file specified` from ninja, not a clear nvcc error.


def _install_fake_cpp_extension(monkeypatch, calls):
    import sys
    import types

    torch_mod = sys.modules.get("torch") or types.ModuleType("torch")
    utils_mod = types.ModuleType("torch.utils")
    cpp_ext_mod = types.ModuleType("torch.utils.cpp_extension")

    def fake_load(**kwargs):
        calls.append(kwargs)
        return object()

    cpp_ext_mod.load = fake_load
    utils_mod.cpp_extension = cpp_ext_mod
    torch_mod.utils = utils_mod

    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "torch.utils", utils_mod)
    monkeypatch.setitem(sys.modules, "torch.utils.cpp_extension", cpp_ext_mod)


def _stub_out_everything_except_check_cl_and_load(monkeypatch, cl_path):
    monkeypatch.setattr(build, "_check_ninja", lambda: None)
    monkeypatch.setattr(build, "_check_cl", lambda: cl_path)
    monkeypatch.setattr(build, "_register_cuda_dll_directory", lambda: None)
    monkeypatch.setattr(build, "_set_cuda_arch_list_if_unset", lambda: None)
    monkeypatch.setattr(build, "_build_directory", lambda: "/fake/build/dir")
    monkeypatch.setattr(build, "_clean_build_directory", lambda d: None)
    monkeypatch.setattr(build.os, "makedirs", lambda *a, **kw: None)


def test_build_passes_ccbin_when_check_cl_resolves_a_path(monkeypatch):
    calls = []
    _install_fake_cpp_extension(monkeypatch, calls)
    _stub_out_everything_except_check_cl_and_load(monkeypatch, r"C:\fake\cl.exe")

    build.build(clean=False)

    assert len(calls) == 1
    cflags = calls[0]["extra_cuda_cflags"]
    assert "-ccbin" in cflags
    assert cflags[cflags.index("-ccbin") + 1] == r"C:\fake\cl.exe"


def test_build_omits_ccbin_when_check_cl_returns_empty(monkeypatch):
    """_check_cl() returns "" on non-Windows (see
    test_check_cl_is_a_noop_on_non_windows) -- build() must not pass a
    bogus empty -ccbin argument to nvcc in that case."""
    calls = []
    _install_fake_cpp_extension(monkeypatch, calls)
    _stub_out_everything_except_check_cl_and_load(monkeypatch, "")

    build.build(clean=False)

    assert len(calls) == 1
    cflags = calls[0]["extra_cuda_cflags"]
    assert "-ccbin" not in cflags


def test_build_still_passes_allow_unsupported_compiler(monkeypatch):
    """The -ccbin fix must not have dropped the existing
    -allow-unsupported-compiler flag it sits alongside."""
    calls = []
    _install_fake_cpp_extension(monkeypatch, calls)
    _stub_out_everything_except_check_cl_and_load(monkeypatch, r"C:\fake\cl.exe")

    build.build(clean=False)

    assert "-allow-unsupported-compiler" in calls[0]["extra_cuda_cflags"]
