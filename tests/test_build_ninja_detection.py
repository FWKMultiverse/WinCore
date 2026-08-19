"""
Tests for the ninja auto-detection/self-heal logic in
WinCore.kernels.build (_find_ninja_binary_dir and _check_ninja) --
added after a real machine hit `RuntimeError: Ninja is required...`
from `_check_ninja()` even though `pip install ninja` had genuinely
been run and `import ninja` worked fine; only `shutil.which("ninja")`
was failing (see build.py's module docstring, failure #7, for the
full story).

Same approach as test_build_msvc_detection.py: everything here is
mocked (shutil.which, sys.modules for the `ninja` import, os.path)
since there's no real ninja-missing-from-PATH machine available to
test against directly -- what's verified is the decision logic.
"""
import os
import sys
import types
import warnings

import pytest

import WinCore.kernels.build as build


def _fake_ninja_module(bin_dir=None, pkg_dir=None, has_bin_dir_attr=True):
    """Build a fake `ninja` module object with just enough shape for
    `_find_ninja_binary_dir` to introspect, without needing the real
    package installed."""
    mod = types.ModuleType("ninja")
    mod.__version__ = "1.13.0-fake"
    mod.__file__ = os.path.join(pkg_dir or "/fake/site-packages/ninja", "__init__.py")
    if has_bin_dir_attr and bin_dir is not None:
        mod.BIN_DIR = bin_dir
    return mod


def test_check_ninja_passes_silently_when_already_on_path(monkeypatch):
    monkeypatch.setattr(build.shutil, "which", lambda name: "/usr/bin/ninja")
    build._check_ninja()  # should not raise, should not even try to import ninja


def test_find_ninja_binary_dir_uses_bin_dir_attribute_when_present(monkeypatch, tmp_path):
    bin_dir = tmp_path / "ninja_bin"
    bin_dir.mkdir()
    exe_name = "ninja.exe" if os.name == "nt" else "ninja"
    (bin_dir / exe_name).write_text("fake binary")

    fake_mod = _fake_ninja_module(bin_dir=str(bin_dir))
    monkeypatch.setitem(sys.modules, "ninja", fake_mod)

    assert build._find_ninja_binary_dir() == str(bin_dir)


def test_find_ninja_binary_dir_falls_back_to_package_dir(monkeypatch, tmp_path):
    pkg_dir = tmp_path / "ninja_pkg"
    pkg_dir.mkdir()
    exe_name = "ninja.exe" if os.name == "nt" else "ninja"
    (pkg_dir / exe_name).write_text("fake binary")

    # No BIN_DIR attribute at all -- older ninja package layout.
    fake_mod = _fake_ninja_module(pkg_dir=str(pkg_dir), has_bin_dir_attr=False)
    monkeypatch.setitem(sys.modules, "ninja", fake_mod)

    assert build._find_ninja_binary_dir() == str(pkg_dir)


def test_find_ninja_binary_dir_falls_back_to_data_bin_subdir(monkeypatch, tmp_path):
    pkg_dir = tmp_path / "ninja_pkg"
    data_bin = pkg_dir / "data" / "bin"
    data_bin.mkdir(parents=True)
    exe_name = "ninja.exe" if os.name == "nt" else "ninja"
    (data_bin / exe_name).write_text("fake binary")

    fake_mod = _fake_ninja_module(pkg_dir=str(pkg_dir), has_bin_dir_attr=False)
    monkeypatch.setitem(sys.modules, "ninja", fake_mod)

    assert build._find_ninja_binary_dir() == str(data_bin)


def test_find_ninja_binary_dir_returns_none_when_package_not_importable(monkeypatch):
    monkeypatch.setitem(sys.modules, "ninja", None)  # forces ImportError on `import ninja`
    assert build._find_ninja_binary_dir() is None


def test_find_ninja_binary_dir_returns_none_when_no_known_layout_matches(monkeypatch, tmp_path):
    # Package importable, but no binary anywhere this function looks --
    # an unrecognized layout, not a "missing" state.
    pkg_dir = tmp_path / "ninja_pkg"
    pkg_dir.mkdir()
    fake_mod = _fake_ninja_module(pkg_dir=str(pkg_dir), has_bin_dir_attr=False)
    monkeypatch.setitem(sys.modules, "ninja", fake_mod)

    assert build._find_ninja_binary_dir() is None


def test_check_ninja_self_heals_by_adding_bin_dir_to_path(monkeypatch, tmp_path):
    """The core fix: ninja isn't on PATH yet shutil.which fails, but the
    importable `ninja` package's binary dir is locatable -- _check_ninja
    should add it to PATH and succeed instead of raising."""
    bin_dir = tmp_path / "ninja_bin"
    bin_dir.mkdir()
    exe_name = "ninja.exe" if os.name == "nt" else "ninja"
    (bin_dir / exe_name).write_text("fake binary")

    fake_mod = _fake_ninja_module(bin_dir=str(bin_dir))
    monkeypatch.setitem(sys.modules, "ninja", fake_mod)

    which_calls = []

    def fake_which(name):
        which_calls.append(name)
        # First call (before self-heal): not found. After PATH is
        # patched to include bin_dir, simulate it now resolving.
        if len(which_calls) == 1:
            return None
        return os.path.join(bin_dir, exe_name)

    monkeypatch.setattr(build.shutil, "which", fake_which)
    monkeypatch.setattr(build.os, "environ", dict(build.os.environ))

    build._check_ninja()  # should not raise
    assert str(bin_dir) in build.os.environ["PATH"]


def test_check_ninja_raises_distinct_message_when_package_installed_but_binary_unreachable(
    monkeypatch, tmp_path
):
    pkg_dir = tmp_path / "ninja_pkg"
    pkg_dir.mkdir()  # no binary anywhere inside -- unrecognized layout
    fake_mod = _fake_ninja_module(pkg_dir=str(pkg_dir), has_bin_dir_attr=False)
    monkeypatch.setitem(sys.modules, "ninja", fake_mod)
    monkeypatch.setattr(build.shutil, "which", lambda name: None)

    with pytest.raises(RuntimeError, match="IS importable"):
        build._check_ninja()


def test_check_ninja_raises_install_instructions_when_package_missing_entirely(monkeypatch):
    monkeypatch.setitem(sys.modules, "ninja", None)
    monkeypatch.setattr(build.shutil, "which", lambda name: None)

    with pytest.raises(RuntimeError, match="pip install ninja"):
        build._check_ninja()
