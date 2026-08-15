"""
Safe torch.compile — Windows-aware defaults + automatic eager fallback.

Why this exists
----------------
torch.compile's default backend (Inductor) generates code through Triton.
Triton has no official Windows support and the maintainers don't accept
community PRs to add it (https://github.com/triton-lang/triton/issues/1640).
Windows users depend on unofficial community forks instead. In practice
this means compiled models can fail *at call time* (not at wrap time,
since torch.compile is lazy — the graph is only built on first real
invocation) with errors like:

    BackendCompilerFailed: backend='inductor' raised:
    PermissionError: [WinError 5] Access is denied: '...\\triton\\...'

This module does two things:
  1. `should_compile()` — defaults to False on Windows, True elsewhere.
     Override with env vars `WINML_SAFE_FORCE_COMPILE=1` or
     `WINML_SAFE_DISABLE_COMPILE=1`.
  2. `safe_compile(module, ...)` — wraps a module with torch.compile, but
     returns a `SafeCompiled` proxy that catches failures on the *first
     real call* and permanently falls back to the eager module for the
     rest of the process. This is the part a bare `try/except` around
     `torch.compile(...)` itself can't catch, because that call never
     actually compiles anything — it just wraps.

Also sets a per-process TORCHINDUCTOR_CACHE_DIR on Windows (if the user
hasn't already set one) to reduce cache-collision odds between concurrent
processes sharing %TEMP%. This does not fully eliminate Windows Triton
issues — that's a limitation of Triton itself — but it removes one
avoidable source of collision.
"""
from __future__ import annotations

import os
import platform
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

if platform.system() == "Windows" and "TORCHINDUCTOR_CACHE_DIR" not in os.environ:
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(
        Path(tempfile.gettempdir()) / f"winml_safe_inductor_{os.getpid()}"
    )


def should_compile() -> bool:
    """Whether torch.compile should be attempted by default on this machine.

    Resolution order:
      1. `WINML_SAFE_DISABLE_COMPILE=1` → always False.
      2. `WINML_SAFE_FORCE_COMPILE=1` → always True (use at your own risk
         on Windows — Triton's Windows support is unofficial).
      3. Otherwise: False on Windows, True elsewhere.
    """
    if _env_flag("WINML_SAFE_DISABLE_COMPILE"):
        return False
    if _env_flag("WINML_SAFE_FORCE_COMPILE"):
        return True
    return platform.system() != "Windows"


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


class SafeCompiled:
    """Callable proxy around a torch.compile()-wrapped module.

    Falls back permanently to the eager module the first time the
    compiled path raises anything at call time. Any attribute access
    beyond calling it (`.parameters()`, `.state_dict()`, `.to()`,
    `.train()`, `.eval()`, custom attributes, ...) is forwarded to the
    underlying eager module via `__getattr__`, so this is a genuine
    drop-in replacement for the wrapped `nn.Module` in a training loop
    -- not only for its forward call. Delegating to the EAGER module
    specifically (not the compiled callable) is deliberate:
    `torch.compile()` optimizes the forward pass, it does not clone the
    module or duplicate its parameters, so `.parameters()`/
    `.state_dict()` return the same, consistent result regardless of
    which path (compiled or eager-fallback) is currently active for
    `__call__`.
    """

    def __init__(self, eager_module: Any, compiled_callable: Any, on_fallback: Optional[Callable[[Exception], None]] = None):
        self._eager = eager_module
        self._compiled = compiled_callable
        self._fallen_back = False
        self._on_fallback = on_fallback

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._fallen_back:
            return self._eager(*args, **kwargs)
        try:
            return self._compiled(*args, **kwargs)
        except Exception as e:
            self._fallen_back = True
            if self._on_fallback is not None:
                self._on_fallback(e)
            return self._eager(*args, **kwargs)

    @property
    def fell_back(self) -> bool:
        return self._fallen_back

    def __getattr__(self, name: str) -> Any:
        # __getattr__ only runs when normal lookup (instance __dict__,
        # class attributes/properties, __call__, etc.) already failed
        # -- so this can't shadow _eager/_compiled/_fallen_back/
        # _on_fallback/fell_back/__call__, all of which resolve
        # normally above. Everything else (nn.Module's own API,
        # user-defined attributes on the wrapped module) forwards to
        # the eager module. Raises the same AttributeError the caller
        # would get from the eager module directly if it truly doesn't
        # have that attribute either -- no swallowing.
        return getattr(self._eager, name)


def safe_compile(
    module: Any,
    mode: Optional[str] = None,
    fullgraph: bool = False,
    enabled: Optional[bool] = None,
    on_fallback: Optional[Callable[[Exception], None]] = None,
    **compile_kwargs: Any,
) -> Any:
    """Wrap `module` with torch.compile, with graceful fallback built in.

    Args:
        module: an nn.Module (or any callable) to compile.
        mode: passed through to torch.compile (e.g. "reduce-overhead").
        fullgraph: passed through to torch.compile.
        enabled: override should_compile() for this call specifically.
            Pass False to force eager (e.g. under a debug flag), True to
            force compilation attempt regardless of platform.
        on_fallback: optional callback invoked once, the first time the
            compiled path fails at runtime — receives the exception. Use
            this to log the fallback instead of silently swallowing it.
        **compile_kwargs: forwarded to torch.compile().

    Returns:
        Either `module` unchanged (compile disabled or torch.compile
        unavailable / failed to wrap), or a `SafeCompiled` proxy that
        behaves like the compiled callable but falls back to eager on
        first runtime failure.
    """
    import torch  # local import: keep this file importable without torch installed

    want_compile = should_compile() if enabled is None else enabled
    if not want_compile or not hasattr(torch, "compile"):
        return module

    try:
        compiled = torch.compile(module, mode=mode, fullgraph=fullgraph, **compile_kwargs)
    except Exception as e:
        if on_fallback is not None:
            on_fallback(e)
        return module

    return SafeCompiled(module, compiled, on_fallback=on_fallback)
