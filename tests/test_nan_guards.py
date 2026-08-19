"""
Tests for WinCore.diagnostics.attach_nan_guards. Requires torch --
skips automatically without it (same pattern as test_fused_bias_gelu.py).
Not run in the sandbox this was written in (no torch there); run on a
machine with torch installed to get real results.
"""
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from WinCore.diagnostics import attach_nan_guards


class _PoisonLayer(nn.Module):
    """Deliberately produces NaN so the guard has something real to
    catch -- not a hidden dependency, just a small test fixture."""

    def forward(self, x):
        return x / 0.0  # -> inf/nan depending on sign


def test_forward_hook_catches_nan_from_specific_layer():
    model = nn.Sequential(nn.Linear(4, 4), _PoisonLayer(), nn.Linear(4, 4))
    seen = []
    guard = attach_nan_guards(model, on_issue=lambda i: seen.append(i), check_backward=False)

    x = torch.randn(2, 4)
    model(x)

    guard.detach()
    assert len(seen) == 1
    assert seen[0].code == "module_output_nan"
    assert "1" in seen[0].data["module"]  # the poison layer's index in the Sequential


def test_clean_model_raises_no_issue():
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    seen = []
    guard = attach_nan_guards(model, on_issue=lambda i: seen.append(i))

    x = torch.randn(3, 4)
    out = model(x)
    out.sum().backward()

    guard.detach()
    assert seen == []


def test_detach_removes_hooks():
    model = nn.Sequential(_PoisonLayer())
    seen = []
    guard = attach_nan_guards(model, on_issue=lambda i: seen.append(i), check_backward=False)
    guard.detach()

    model(torch.randn(2, 4))  # would trigger if hooks were still attached
    assert seen == []


def test_raise_on_detect_raises_floatingpointerror():
    model = nn.Sequential(_PoisonLayer())
    guard = attach_nan_guards(model, raise_on_detect=True, check_backward=False)
    with pytest.raises(FloatingPointError):
        model(torch.randn(2, 4))
    guard.detach()


def test_context_manager_detaches_on_exit():
    model = nn.Sequential(_PoisonLayer())
    seen = []
    with attach_nan_guards(model, on_issue=lambda i: seen.append(i), check_backward=False) as guard:
        model(torch.randn(2, 4))
    assert len(seen) == 1

    seen.clear()
    model(torch.randn(2, 4))  # hooks detached now -- no new issue
    assert seen == []
