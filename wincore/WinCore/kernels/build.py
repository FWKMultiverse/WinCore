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
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Optional


def _check_ninja() -> None:
    if shutil.which("ninja") is None:
        raise RuntimeError(
            "Ninja is required to build this extension but wasn't found on PATH.\n"
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
    hard crash."""
    try:
        result = subprocess.run(
            [
                vswhere_path, "-latest", "-products", "*",
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
        "making sure the 'MSVC v143 - VS 2022 C++ x64/x86 build tools' "
        "component is checked, then re-run this -- auto-detection will find "
        "it next time without any special terminal.\n\n"
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
    """A build directory WE own and control fully, so `clean=True` is
    guaranteed to wipe exactly what `load()` will use -- not relying on
    torch's private default-directory logic, which turned out not to
    match reliably (see failure #2 in the module docstring)."""
    return os.path.join(tempfile.gettempdir(), "wincore_build_fused_bias_gelu")


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

    if clean and os.path.isdir(build_dir):
        shutil.rmtree(build_dir, ignore_errors=True)
    os.makedirs(build_dir, exist_ok=True)

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
