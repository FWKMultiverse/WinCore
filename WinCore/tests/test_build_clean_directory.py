"""
Tests for _clean_build_directory in WinCore.kernels.build -- added
after discovering that the old `shutil.rmtree(build_dir,
ignore_errors=True)` silently swallowed a real Windows failure mode:
once a compiled .pyd from a previous build() call is loaded into the
current process, Windows locks that file and rmtree can't remove it.
`ignore_errors=True` hid that completely, so `clean=True` would claim
success while leaving the stale extension in place -- confirmed on a
real machine via the "No modifications detected ... skipping build
step" log line appearing on a second build() call in the same
process (see build.py's module docstring for the full story).
"""
import os
import warnings

import WinCore.kernels.build as build


def test_clean_build_directory_removes_a_normal_directory(tmp_path):
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    (build_dir / "stale.o").write_text("old object file")

    build._clean_build_directory(str(build_dir))

    assert not build_dir.exists()


def test_clean_build_directory_is_a_noop_when_directory_does_not_exist(tmp_path):
    build_dir = tmp_path / "does_not_exist"
    build._clean_build_directory(str(build_dir))  # should not raise


def test_clean_build_directory_warns_when_a_file_cannot_be_removed(tmp_path, monkeypatch):
    """Simulates the real Windows scenario: a .pyd from a previously
    loaded extension is locked and can't be deleted. shutil.rmtree is
    mocked to report exactly that failure via onerror, the way it
    would for a real locked file -- this verifies _clean_build_directory
    surfaces it as a warning instead of silently ignoring it."""
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    locked_file = build_dir / "wincore_fused_bias_gelu.pyd"
    locked_file.write_text("pretend this is a loaded, locked .pyd")

    def fake_rmtree(path, onerror=None, ignore_errors=False):
        if onerror is not None:
            try:
                raise PermissionError("simulated: file in use by another process")
            except PermissionError as e:
                onerror(os.remove, str(locked_file), (type(e), e, e.__traceback__))
        # directory deliberately left in place, mirroring a real
        # partial-failure rmtree that couldn't remove the locked file

    monkeypatch.setattr(build.shutil, "rmtree", fake_rmtree)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build._clean_build_directory(str(build_dir))
    matched = [w for w in caught if issubclass(w.category, RuntimeWarning)
               and "couldn't fully remove" in str(w.message)]
    assert matched, f"expected a RuntimeWarning about the locked file, got: {caught}"


def test_clean_build_directory_does_not_warn_on_full_success(tmp_path, monkeypatch):
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    (build_dir / "ok.o").write_text("fine")

    real_rmtree = build.shutil.rmtree  # capture before patching -- build.shutil IS
    # the real shutil module, so patching build.shutil.rmtree mutates the
    # global shutil.rmtree too; grab the original first or this recurses.

    def fake_rmtree(path, onerror=None, ignore_errors=False):
        real_rmtree(path)  # actually succeeds, no onerror calls

    monkeypatch.setattr(build.shutil, "rmtree", fake_rmtree)

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning here should fail the test
        build._clean_build_directory(str(build_dir))
