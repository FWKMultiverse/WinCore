"""
Compile the fused_bias_gelu CUDA extension.

Requirements (Windows):
  - `pip install ninja` (separate pip package, not bundled with torch).
  - NVIDIA CUDA Toolkit installed, matching your installed torch build's
    CUDA version (`python -c "import torch; print(torch.version.cuda)"`).
  - MSVC's `cl.exe` reachable on PATH -- see "Real failure #3" below,
    this is almost certainly what you're hitting if you're reading this
    after a build error.
  - A CUDA-capable GPU with compute capability >= 6.0.

Usage:
    python -m WinCore.kernels.build

Real failures hit on an actual Windows machine, in order, and the fix
applied for each
-----------------------------------------------------------------------
1. `RuntimeError: Ninja is required to load C++ extensions` -> fixed:
   `_check_ninja()` checks upfront with an actionable message instead
   of a cryptic error from deep in torch internals.

2. `ImportError: DLL load failed ... The specified module could not be
   found`, after a log line `"No modifications detected ... skipping
   build step"` -> a stale/partial build artifact from a previous
   failed attempt was being silently reused instead of rebuilt. Fixed
   by managing an explicit `build_directory` ourselves (not relying on
   torch's private default-directory logic, which turned out not to be
   the same directory `load()` actually used -- the previous version of
   this file cleaned the wrong/mismatched directory) and deleting it
   before every build when `clean=True` (the default).

3. `subprocess.CalledProcessError: Command '['where', 'cl']' returned
   non-zero exit status 1`, alongside a warning `Error checking
   compiler version for cl: [WinError 2] The system cannot find the
   file specified` -> this was an environment problem, not something
   this script can silently fix: MSVC's `cl.exe` was not on PATH in the
   shell pytest/python was running in. A plain PowerShell +
   `Activate.ps1` does NOT set up the MSVC compiler environment on its
   own. `_check_cl()` now checks for this upfront with the actionable
   fix spelled out (see its docstring / the RuntimeError message). Real
   fix confirmed: opening the "Developer Command Prompt for VS" (which
   runs vcvarsall.bat) before activating the venv resolved this.

4. `fatal error C1189: -- unsupported Microsoft Visual Studio version!
   Only the versions between 2019 and 2022 (inclusive) are supported!`
   -> confirmed on a real machine running a very new Visual Studio 2026
   alongside CUDA Toolkit 13.1: nvcc's own host-compiler compatibility
   list hasn't been updated for VS2026 yet (CUDA Toolkit releases
   always lag behind the newest VS release by some months). nvcc's own
   error message names the fix -- the `-allow-unsupported-compiler`
   flag -- now added to `extra_cuda_cflags`. NVIDIA's own warning on
   that flag applies here too: skipping the check means an
   officially-untested compiler combination is being used, which
   *usually* works fine for straightforward kernel code like this one
   (no exotic C++ features), but isn't guaranteed -- if you see wrong
   numerical results (not just a build failure) after this, that flag
   combined with VS2026 would be the first thing to suspect, and
   installing VS2022 Build Tools side-by-side (keeping VS2026 as your
   main IDE) as the compiler actually used for this specific build
   would be the safer long-term fix.

None of these four fixes had been re-verified end-to-end against a
real Windows+CUDA+cl.exe machine as of the previous edit to this file
-- see #6 below for the real machine that finally confirmed them.

5. Needing to open 'x64 Native Tools Command Prompt for VS'
   specifically (a plain PowerShell/cmd, or an IDE-integrated terminal
   in VSCode/Cursor/PyCharm/etc, doesn't have cl.exe on PATH) was real
   friction reported when trying to get this running from an IDE
   directly instead of a special launcher. `_check_cl()` now tries
   `_autodetect_and_setup_msvc()` first: it locates the VS install via
   vswhere.exe (Microsoft's own supported lookup, same fixed path any
   VS Installer places it at) and vcvars64.bat under it, runs that
   script, and imports the PATH/INCLUDE/LIB/LIBPATH variables it
   produces into THIS process -- so `cl.exe` becomes findable without
   ever needing a different terminal. Only falls through to the manual
   "open that special prompt" instructions if auto-detection genuinely
   can't find a working MSVC install at all (this is the fallback, not
   the primary path anymore).

6. First confirmed end-to-end success on a real Windows + CUDA + MSVC
   machine (VS2026 host compiler via `-allow-unsupported-compiler`,
   torch's own CUDA build): all 11 tests in
   `tests/test_fused_bias_gelu.py` passed, including the timing
   comparison against the unfused fallback -- so the fixes for
   failures #1-5 above are now verified end-to-end, not just
   individually unit-tested. Two warnings appeared alongside the pass,
   neither of which indicated a real problem:
     - `TORCH_CUDA_ARCH_LIST is not set, all archs ... included` --
       real and now fixed: `_set_cuda_arch_list_if_unset()` detects the
       installed GPU(s) and sets it automatically (unless the caller
       already set it), so the build only targets hardware actually
       present.
     - `_get_vc_env is private; find an alternative` -- comes from
       inside `setuptools`/`torch.utils.cpp_extension`'s own MSVC
       detection, not from any WinCore code; nothing here calls
       `_get_vc_env`. Not fixable from this module, and does not affect
       the build result -- an upstream cosmetic warning to expect and
       ignore.

7. `RuntimeError: Ninja is required to build this extension but wasn't
   found on PATH` even with `ninja==1.13.0` genuinely `pip install`ed
   -- confirmed on a real machine: `shutil.which("ninja")` failed
   inside a pytest subprocess even though the identical check
   succeeded moments later in a different process on the same
   machine, same venv, same PATH the shell should have had. The old
   `_check_ninja()` only ever tried `shutil.which` once and gave up.
   Fixed the same way `_check_cl` already handles the equivalent MSVC
   problem: `_find_ninja_binary_dir()` locates the binary directly
   from the importable `ninja` Python package (via `ninja.BIN_DIR` on
   newer releases, or known fallback layouts on older ones) and
   self-heals by adding that directory to this process's PATH,
   instead of failing a build that could have actually worked. The
   error message now also distinguishes "ninja isn't installed at
   all" from "ninja IS installed but its binary wasn't findable in
   any known layout" -- those need different fixes and were
   previously indistinguishable from one error string.

8. `_check_cl()`'s auto-detection (vswhere + vcvars64.bat) found
   nothing at all on a real machine that DOES have Visual Studio 2026
   (MSVC v145) with the C++ workload installed -- silently fell all
   the way through to the manual "install Build Tools" instructions,
   which was actively wrong advice since Build Tools were already
   there. Root cause: `vswhere -latest ...` without `-prerelease`
   only considers STABLE-channel VS installs, and VS2026 was still on
   the Preview/Insider channel at the time. `_find_vcvars64()` now
   passes `-prerelease`, which is a strict superset of the previous
   query (can only find MORE installs, never fewer) -- confirmed this
   finds the VS2026 install and vcvars64.bat loads correctly from it.
   The error message in `_check_cl()` also no longer names one
   specific toolset ("MSVC v143 - VS 2022") as if it were the only one
   that works, since v143 through v145 have now all been confirmed
   fine for this kernel.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import warnings
from typing import Optional


def _find_ninja_binary_dir() -> Optional[str]:
    """`pip install ninja` (the scikit-build-maintained PyPI package,
    not a system package) bundles a prebuilt `ninja` binary INSIDE the
    Python package -- it's supposed to end up on PATH automatically via
    a console-script shim in <venv>/Scripts, but that shim depends on
    <venv>/Scripts already being on PATH at the moment pip installed
    it. Confirmed on a real machine: `shutil.which("ninja")` returned
    None inside a pytest subprocess even though `ninja==1.13.0` was
    genuinely pip-installed and importable, and even though the exact
    same check succeeded moments later in a different process on the
    same machine -- i.e. this is a real, intermittent PATH-resolution
    gap, not a "ninja isn't installed" situation, and the old
    single-shot `shutil.which` check couldn't tell the two apart or
    recover from the first one.

    This tries to locate the binary directly from the importable
    Python package instead of relying on PATH alone: recent `ninja`
    releases expose `ninja.BIN_DIR` specifically for this purpose;
    older ones bundle it next to `ninja/__init__.py` or under a
    `ninja/data/bin` subdirectory. Returns the directory containing a
    `ninja`/`ninja.exe` file, or None if the package isn't importable
    or none of the known layouts match (in which case ninja really
    isn't usable and the actionable error below is accurate, not a
    false negative)."""
    try:
        import ninja  # noqa: PLC0415 -- deliberately lazy, ninja is optional
    except ImportError:
        return None

    exe_name = "ninja.exe" if os.name == "nt" else "ninja"
    candidates = []
    bin_dir = getattr(ninja, "BIN_DIR", None)
    if bin_dir:
        candidates.append(bin_dir)
    pkg_dir = os.path.dirname(os.path.abspath(ninja.__file__))
    candidates.append(pkg_dir)
    candidates.append(os.path.join(pkg_dir, "data", "bin"))

    for d in candidates:
        if d and os.path.isfile(os.path.join(d, exe_name)):
            return d
    return None


def _check_ninja() -> None:
    if shutil.which("ninja") is not None:
        return
    # Not on PATH -- before concluding it's missing, check whether the
    # `ninja` pip package is actually installed and just isn't wired
    # onto PATH yet (see `_find_ninja_binary_dir` docstring for why
    # this genuinely happens). If so, self-heal the same way
    # `_autodetect_and_setup_msvc` does for cl.exe: add the directory
    # to THIS process's PATH and move on, rather than failing a build
    # that could have just worked.
    bin_dir = _find_ninja_binary_dir()
    if bin_dir is not None:
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
        if shutil.which("ninja") is not None:
            return
    # Genuinely missing (package not importable) vs. installed-but-
    # unreachable (package importable, binary not found in any known
    # layout) are different problems with different fixes -- report
    # which one this actually is instead of one generic message.
    try:
        import ninja as _ninja_probe
        installed_but_unreachable = (
            f"the `ninja` package IS importable (version "
            f"{getattr(_ninja_probe, '__version__', 'unknown')}, at "
            f"{os.path.dirname(os.path.abspath(_ninja_probe.__file__))}) but its "
            f"bundled binary wasn't found in any of the locations this checks "
            f"(BIN_DIR, package dir, package dir/data/bin) -- this looks like an "
            f"unusual `ninja` package layout this hasn't seen before. Try `pip "
            f"install --force-reinstall ninja`, or install the standalone Ninja "
            f"build tool and put it on PATH yourself."
        )
        raise RuntimeError(installed_but_unreachable)
    except ImportError:
        pass
    raise RuntimeError(
        "Ninja is required to build this extension but wasn't found on PATH, "
        "and the `ninja` Python package isn't installed either.\n"
        "Install it with:\n\n    pip install ninja\n\n"
        "then re-run `python -m WinCore.kernels.build`."
    )


def _find_vswhere() -> Optional[str]:
    """vswhere.exe ships with the Visual Studio Installer since VS2017
    and lives at one of these two fixed, well-known paths regardless of
    which VS edition/version is actually installed -- this is
    Microsoft's own supported way to locate a VS installation
    programmatically (the same mechanism CMake's FindVisualStudio and
    many other build tools use), not something guessed at."""
    for env_var in ("ProgramFiles(x86)", "ProgramFiles"):
        base = os.environ.get(env_var)
        if not base:
            continue
        candidate = os.path.join(base, "Microsoft Visual Studio", "Installer", "vswhere.exe")
        if os.path.isfile(candidate):
            return candidate
    return None


def _find_vcvars64(vswhere_path: str) -> Optional[str]:
    """Asks vswhere for the latest VS install that has the C++ x86/x64
    build tools component, then looks for vcvars64.bat under it --
    the same script 'x64 Native Tools Command Prompt for VS' itself
    runs on startup. Returns None (not an exception) on any failure --
    every failure mode here just means auto-detection didn't work,
    which falls back to the manual instructions in `_check_cl`, not a
    hard crash.

    Includes `-prerelease`: without it, vswhere only returns STABLE-
    channel installs. Confirmed on a real machine running Visual
    Studio 2026 (MSVC v145) -- while that release was still on VS's
    Preview/Insider channel, a plain `vswhere -latest ...` query (no
    `-prerelease`) returned nothing at all for it, even with the C++
    workload fully installed, so detection silently fell all the way
    through to the manual-fix error message. `-prerelease` makes
    vswhere consider preview/insider instances too -- it's a strict
    superset of what the query found before, so this can't cause a
    previously-found stable install to stop being found."""
    try:
        result = subprocess.run(
            [
                vswhere_path, "-latest", "-prerelease", "-products", "*",
                "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property", "installationPath",
            ],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired, subprocess.SubprocessError):
        return None
    install_path = result.stdout.strip()
    if not install_path:
        return None
    candidate = os.path.join(install_path, "VC", "Auxiliary", "Build", "vcvars64.bat")
    return candidate if os.path.isfile(candidate) else None


_MSVC_ENV_VARS_TO_IMPORT = ("PATH", "INCLUDE", "LIB", "LIBPATH")


def _apply_vcvars64_env(vcvars64_path: str) -> bool:
    """Runs vcvars64.bat in a child cmd.exe, dumps the resulting
    environment with `set`, and merges PATH/INCLUDE/LIB/LIBPATH into
    THIS process's os.environ -- so cl.exe becomes findable for the
    rest of this Python process without the user needing a different
    terminal at all. This is the same technique setuptools' own MSVC
    support (`distutils._msvccompiler`) and other build tools use to
    pick up the MSVC environment programmatically instead of requiring
    a specific launcher shell.

    Only PATH/INCLUDE/LIB/LIBPATH are imported -- not the full `set`
    dump -- deliberately: vcvars64.bat's own output includes plenty of
    VS-internal bookkeeping variables that this process has no reason
    to inherit; these four are the ones cl.exe/nvcc actually need to
    find the compiler, headers, and libraries."""
    try:
        result = subprocess.run(
            f'cmd.exe /c ""{vcvars64_path}" && set"',
            capture_output=True, text=True, shell=True, timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired, subprocess.SubprocessError):
        return False
    if result.returncode != 0:
        return False

    applied = False
    for line in result.stdout.splitlines():
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip().upper() in _MSVC_ENV_VARS_TO_IMPORT:
            os.environ[key.strip().upper()] = value
            applied = True
    return applied


def _autodetect_and_setup_msvc() -> bool:
    """Best-effort, fully automatic: if `cl.exe` isn't already on PATH,
    try to locate and load the MSVC compiler environment via vswhere +
    vcvars64.bat, so a plain terminal or IDE-integrated terminal
    (VSCode, Cursor, PyCharm, a regular PowerShell window, ...) works
    without the user needing to specifically open 'x64 Native Tools
    Command Prompt for VS' first. Returns True if `cl.exe` is findable
    afterward (either it already was, or auto-setup just made it so).
    Every internal step degrades to `False` on failure rather than
    raising -- `_check_cl()` below is what turns a `False` here into
    the actionable manual-fix error message, so this function itself
    never needs to explain anything to the user."""
    if shutil.which("cl") is not None:
        return True
    if os.name != "nt":
        return False
    vswhere = _find_vswhere()
    if vswhere is None:
        return False
    vcvars64 = _find_vcvars64(vswhere)
    if vcvars64 is None:
        return False
    if not _apply_vcvars64_env(vcvars64):
        return False
    return shutil.which("cl") is not None


def _check_cl() -> None:
    """MSVC's cl.exe is required. Tries automatic detection first (see
    `_autodetect_and_setup_msvc`) -- only falls through to the manual
    instructions below if that genuinely couldn't find/apply a working
    MSVC environment (e.g. Build Tools aren't installed at all, or
    vswhere/vcvars64.bat aren't where expected)."""
    if os.name != "nt":
        return
    if _autodetect_and_setup_msvc():
        return
    raise RuntimeError(
        "cl.exe (the MSVC compiler) was not found on PATH, and automatic "
        "detection (via vswhere.exe + vcvars64.bat) couldn't locate or load "
        "a working MSVC environment either -- most likely the C++ Build "
        "Tools component itself isn't installed. Two ways to fix this:\n\n"
        "  1. Install 'Desktop development with C++' via the Visual Studio "
        "Installer (or just the standalone 'Build Tools for Visual Studio'), "
        "making sure a 'C++ x64/x86 build tools' component is checked "
        "(any recent MSVC toolset works -- v143 through v145 have all been "
        "confirmed; auto-detection includes preview/Insider VS channels "
        "too, not just stable), then re-run this -- auto-detection will "
        "find it next time without any special terminal.\n\n"
        "  2. If Build Tools ARE installed but auto-detection still isn't "
        "finding them (e.g. an unusual install location), open 'x64 Native "
        "Tools Command Prompt for VS' from the Start Menu manually, "
        "activate your .venv inside THAT prompt, then re-run -- this always "
        "works as a fallback since it's the same environment auto-detection "
        "was trying to replicate.\n\n"
        "This is a one-time environment/install issue, not a WinCore or "
        "CUDA problem."
    )


def _register_cuda_dll_directory() -> None:
    """Make the CUDA Toolkit's own bin/ (where the runtime DLLs the
    compiled extension links against actually live) discoverable,
    separately from whatever DLLs ship inside the torch package."""
    if os.name != "nt":
        return
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        bin_dir = os.path.join(cuda_path, "bin")
        if os.path.isdir(bin_dir):
            try:
                os.add_dll_directory(bin_dir)  # py3.8+
            except (AttributeError, OSError):
                pass


def _build_directory() -> str:
    """A build directory WE own and control fully, so `clean=True` can
    target exactly what `load()` will use -- not relying on torch's
    private default-directory logic, which turned out not to match
    reliably (see failure #2 in the module docstring). NOTE: "control
    fully" here means we know which directory to clean, not that
    cleaning it is guaranteed to succeed -- see `_clean_build_directory`
    for the real (Windows-file-lock) limit on that guarantee."""
    return os.path.join(tempfile.gettempdir(), "wincore_build_fused_bias_gelu")


def _clean_build_directory(build_dir: str) -> None:
    """Delete `build_dir` before a rebuild -- but, unlike the previous
    `shutil.rmtree(build_dir, ignore_errors=True)`, actually notice and
    report it when that deletion doesn't fully succeed, instead of
    silently pretending it did.

    Why this matters specifically here: once `torch.utils.cpp_extension
    .load()` has loaded the compiled `.pyd` into THIS process, Windows
    holds an OS-level lock on that file for as long as the process has
    it loaded -- unlike a source file, which can still be edited or
    deleted while open. `ignore_errors=True` swallowed that
    `PermissionError` completely. The observable symptom (confirmed on
    a real machine): call `build()` once, then call it again later in
    the SAME process (e.g. editing `fused_bias_gelu_kernel.cu` and
    rebuilding without restarting Python, or simply two callers
    unaware they're sharing a process, as happened when the
    "kernel_status()" check and the "correctness" check both ran in
    one diagnostic script) -- the second call logs "No modifications
    detected ... skipping build step" and returns the SAME pre-edit
    kernel, even though `clean=True` was requested and should have
    forced a full rebuild. No error was raised anywhere in that
    sequence; it just silently wasn't a rebuild.

    This can't fix the underlying OS constraint -- a currently-loaded
    native extension genuinely cannot be replaced without restarting
    the process -- but it can (and now does) tell you that's what's
    happening instead of pretending the clean succeeded.
    """
    if not os.path.isdir(build_dir):
        return
    locked_paths = []

    def _onerror(_func, path, exc_info):
        locked_paths.append((path, exc_info[1]))

    shutil.rmtree(build_dir, onerror=_onerror)
    if locked_paths and os.path.isdir(build_dir):
        culprit = locked_paths[0][0]
        warnings.warn(
            f"build(clean=True) couldn't fully remove the previous build "
            f"directory ({build_dir}) -- {len(locked_paths)} file(s) are "
            f"still in use, most likely {os.path.basename(culprit)} from an "
            f"extension already loaded earlier in THIS process (Windows "
            f"locks a .pyd/.dll once it's loaded). The upcoming build may "
            f"silently return that already-loaded extension unchanged -- "
            f"including if you just edited fused_bias_gelu_kernel.cu -- "
            f"rather than actually rebuilding it, with no further error to "
            f"tell you so. If you're iterating on the kernel source, "
            f"restart the Python process between builds; a loaded native "
            f"extension can't be hot-swapped.",
            RuntimeWarning,
            stacklevel=3,
        )


def _set_cuda_arch_list_if_unset() -> None:
    """Without TORCH_CUDA_ARCH_LIST set, torch's cpp_extension compiles
    for every arch it knows about (a real warning seen on a real build:
    "TORCH_CUDA_ARCH_LIST is not set, all archs for visible cards are
    included for compilation") -- slower to build and produces a larger
    binary than needed for a single-machine build. This detects the
    actual GPU(s) present via torch and sets it to just those compute
    capabilities, so the build only targets hardware that's actually
    here. Does nothing (respects the override) if the caller already
    set TORCH_CUDA_ARCH_LIST themselves -- e.g. building for a
    different/multiple target machines than the one doing the build."""
    if "TORCH_CUDA_ARCH_LIST" in os.environ:
        return
    try:
        import torch

        if not torch.cuda.is_available():
            return
        caps = {
            "%d.%d" % torch.cuda.get_device_capability(i)
            for i in range(torch.cuda.device_count())
        }
        if caps:
            os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(sorted(caps))
    except Exception:
        # Detection failing here just means the upstream default
        # (build for every arch) applies, same as before this
        # function existed -- not worth failing the build over.
        pass


def build(clean: bool = True):
    _check_ninja()
    _check_cl()
    _register_cuda_dll_directory()
    _set_cuda_arch_list_if_unset()

    from torch.utils.cpp_extension import load

    this_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = _build_directory()

    if clean:
        _clean_build_directory(build_dir)
    os.makedirs(build_dir, exist_ok=True)

    # `_get_vc_env is private; find an alternative` comes from inside
    # setuptools/torch's own MSVC detection (see failure #6 above) --
    # confirmed harmless and not something this module can fix, so it
    # only adds noise to `verbose=True`'s already-long build log.
    # Filtered narrowly by message text, not category, so an
    # unrelated UserWarning from the actual compile still surfaces.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r".*_get_vc_env is private.*", category=UserWarning
        )
        module = load(
            name="wincore_fused_bias_gelu",
            sources=[os.path.join(this_dir, "fused_bias_gelu_kernel.cu")],
            extra_cuda_cflags=["-O3", "-allow-unsupported-compiler"],
            build_directory=build_dir,
            verbose=True,
        )
    return module


if __name__ == "__main__":
    build()
    print("Built OK.")
