"""
Tests for WinCore.compile.SafeCompiled.

Deliberately does NOT import torch or require a GPU: SafeCompiled's
proxying behavior (attribute delegation, fallback-on-failure, the
fell_back property) is plain Python and fully testable with a stand-in
"module" object that mimics the bits of nn.Module's API this proxy
needs to forward -- __call__, .parameters(), .state_dict(), .train(),
.to(). This file exists because that proxying behavior previously had
zero test coverage, which is exactly how two real bugs shipped
unnoticed: `fell_back` being called as `fell_back()` instead of
accessed as the property it actually is, and SafeCompiled having no
attribute delegation at all (so `.parameters()`, `.state_dict()`,
`.train()`, `.to()` all raised AttributeError on a real training loop,
even though calling it directly for a forward pass worked fine).
"""
from WinCore.compile import SafeCompiled


class _FakeModule:
    """Stands in for an nn.Module without needing torch installed."""

    def __init__(self):
        self.mode = None
        self.device = None

    def __call__(self, x):
        return x * 2

    def parameters(self):
        return ["p1", "p2"]

    def state_dict(self):
        return {"w": 1}

    def train(self):
        self.mode = "train"
        return self

    def eval(self):
        self.mode = "eval"
        return self

    def to(self, device):
        self.device = device
        return self


def _always_fails(*args, **kwargs):
    raise RuntimeError("simulated compiled-path failure")


def test_fell_back_is_a_property_not_a_method():
    sc = SafeCompiled(_FakeModule(), lambda x: x)
    assert sc.fell_back is False
    # Calling it like a method must fail the same way calling any bool
    # would -- this is the exact failure mode reported against the
    # (incorrect) API_REFERENCE.md docs, confirming the property
    # contract instead of a callable one.
    try:
        sc.fell_back()
        assert False, "fell_back should not be callable"
    except TypeError:
        pass


def test_successful_compiled_call_does_not_fall_back():
    sc = SafeCompiled(_FakeModule(), lambda x: x * 100)
    assert sc(2) == 200
    assert sc.fell_back is False


def test_forward_falls_back_on_compiled_failure():
    eager = _FakeModule()
    sc = SafeCompiled(eager, _always_fails)
    assert sc(21) == 42          # eager path: x * 2
    assert sc.fell_back is True
    # stays fallen back for subsequent calls too
    assert sc(5) == 10
    assert sc.fell_back is True


def test_on_fallback_callback_invoked_once_with_exception():
    seen = []
    sc = SafeCompiled(_FakeModule(), _always_fails, on_fallback=lambda e: seen.append(e))
    sc(1)
    sc(2)
    assert len(seen) == 1
    assert isinstance(seen[0], RuntimeError)


def test_attribute_delegation_parameters_and_state_dict():
    eager = _FakeModule()
    sc = SafeCompiled(eager, lambda x: x)
    assert sc.parameters() == ["p1", "p2"]
    assert sc.state_dict() == {"w": 1}


def test_attribute_delegation_train_eval_to_mutate_eager_module():
    eager = _FakeModule()
    sc = SafeCompiled(eager, lambda x: x)
    sc.train()
    assert eager.mode == "train"
    sc.eval()
    assert eager.mode == "eval"
    sc.to("cuda:0")
    assert eager.device == "cuda:0"


def test_unknown_attribute_still_raises_attributeerror():
    sc = SafeCompiled(_FakeModule(), lambda x: x)
    try:
        sc.this_attribute_does_not_exist_anywhere
        assert False, "expected AttributeError"
    except AttributeError:
        pass
