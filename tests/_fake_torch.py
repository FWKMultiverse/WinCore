"""
Minimal fake `torch` built on numpy, JUST so the actual WinCore code
paths (kv.py append/replace/eviction, precision.py fp8 quantize logic)
can be *executed* in this sandbox (no real torch/GPU/network available)
instead of only read by eye. This is not a claim that it validates real
fp8 hardware precision loss -- numpy has no float8 dtype, so casts here
keep full precision. What it DOES catch for real: shape bugs, off-by-
one eviction bugs, wrong-axis concatenation, crashes on edge cases
(zero tensors, single updates, invalid modes), scale-factor arithmetic
errors -- i.e. actual logic bugs, not hand-waved as "traced by eye".
"""
import sys
import types
import numpy as np


class FakeDtype:
    def __init__(self, name):
        self.name = name
    def __repr__(self):
        return f"torch.{self.name}"
    def __eq__(self, other):
        return isinstance(other, FakeDtype) and other.name == self.name
    def __hash__(self):
        return hash(self.name)


float16 = FakeDtype("float16")
float32 = FakeDtype("float32")
float64 = FakeDtype("float64")
bfloat16 = FakeDtype("bfloat16")
float8_e4m3fn = FakeDtype("float8_e4m3fn")
float8_e5m2 = FakeDtype("float8_e5m2")


class FakeTensor:
    def __init__(self, arr, dtype=float32):
        self.arr = np.asarray(arr, dtype=np.float64)
        self.dtype = dtype

    @property
    def shape(self):
        return self.arr.shape

    def numel(self):
        return self.arr.size

    def abs(self):
        return FakeTensor(np.abs(self.arr), self.dtype)

    def amax(self, dim=None, keepdim=False):
        if dim is None:
            return FakeTensor(np.array(self.arr.max()), self.dtype)
        axis = tuple(dim) if isinstance(dim, (list, tuple)) else dim
        return FakeTensor(np.max(self.arr, axis=axis, keepdims=keepdim), self.dtype)

    def item(self):
        return float(self.arr)

    def detach(self):
        return self

    def float(self):
        return FakeTensor(self.arr.astype(np.float64), float32)

    def to(self, dtype):
        # Real torch would lose mantissa bits here for fp8; numpy has
        # no float8 dtype so this keeps full precision -- flagged
        # clearly in the report, this is the one thing NOT verified.
        return FakeTensor(self.arr.copy(), dtype)

    def narrow(self, dim, start, length):
        sl = [slice(None)] * self.arr.ndim
        sl[dim] = slice(start, start + length)
        return FakeTensor(self.arr[tuple(sl)], self.dtype)

    def flatten(self):
        return FakeTensor(self.arr.flatten(), self.dtype)

    def reshape(self, shape):
        return FakeTensor(self.arr.reshape(shape), self.dtype)

    def clamp(self, min=None, max=None):
        return FakeTensor(np.clip(self.arr, min, max), self.dtype)

    def tolist(self):
        return self.arr.tolist()

    def __sub__(self, other):
        return FakeTensor(self.arr - other.arr, self.dtype)

    def __add__(self, other):
        # Needed for precision.py's per-channel fp8 quantize path,
        # which adds a tiny epsilon to a per-channel scale tensor to
        # guard an all-zero channel from a 0/0 -> NaN division,
        # instead of the global-scale path's scalar branch-and-replace
        # (which doesn't broadcast the same way over a multi-element
        # scale tensor).
        if isinstance(other, FakeTensor):
            return FakeTensor(self.arr + other.arr, self.dtype)
        return FakeTensor(self.arr + other, self.dtype)

    __radd__ = __add__

    def __truediv__(self, other):
        if isinstance(other, FakeTensor):
            return FakeTensor(self.arr / other.arr, self.dtype)
        return FakeTensor(self.arr / other, self.dtype)

    def __mul__(self, other):
        if isinstance(other, FakeTensor):
            return FakeTensor(self.arr * other.arr, self.dtype)
        return FakeTensor(self.arr * other, self.dtype)

    __rmul__ = __mul__

    def mean(self):
        return FakeTensor(np.array(self.arr.mean()), self.dtype)

    def __eq__(self, other):
        if isinstance(other, FakeTensor):
            return FakeTensor((self.arr == other.arr).astype(float), self.dtype)
        return FakeTensor((self.arr == other).astype(float), self.dtype)

    def all(self):
        return bool(self.arr.astype(bool).all())

    def __bool__(self):
        # Real torch: a 0-d (scalar) tensor implicitly converts to bool
        # via its single value; multi-element tensors raise. This fake
        # was missing this method entirely, which meant `if fake_tensor
        # == 0:` was always truthy (Python's default object truthiness)
        # regardless of the actual comparison result -- a shim bug that
        # silently made quantize_fp8's `amax == 0` branch always fire.
        if self.arr.size != 1:
            raise ValueError("ambiguous truth value of tensor with more than one element")
        return bool(self.arr.item())

    def __repr__(self):
        return f"FakeTensor(shape={self.shape}, dtype={self.dtype})"

    def __getitem__(self, idx):
        # Needed so tests can slice results the way real torch.Tensor
        # supports (e.g. result[:, :, :1]) -- without this, StepCache
        # tests that check *which* slice of an appended tensor came
        # from which update() call can't actually be written.
        return FakeTensor(self.arr[idx], self.dtype)


def cat(tensors, dim=0):
    return FakeTensor(np.concatenate([t.arr for t in tensors], axis=dim), tensors[0].dtype)


def tensor(data, dtype=None):
    return FakeTensor(np.asarray(data, dtype=np.float64), dtype if dtype is not None else float32)


def isfinite(t):
    return bool(np.isfinite(t.arr).all())


def all(t):  # noqa: A001 - matches torch.all's name
    return bool(np.asarray(t.arr).astype(_np_bool()).all())


def _np_bool():
    return np.bool_


def randn(*shape):
    return FakeTensor(np.random.randn(*shape), float32)


def zeros(*shape):
    return FakeTensor(np.zeros(shape), float32)


def full(shape, val):
    return FakeTensor(np.full(shape, val), float32)


def ones(*shape):
    return FakeTensor(np.ones(shape), float32)


def allclose(a, b, atol=1e-6):
    return bool(np.allclose(a.arr, b.arr, atol=atol))


class _FakeGradScaler:
    """Just enough of torch.amp.GradScaler / torch.cuda.amp.GradScaler
    for WinCore.precision.amp()'s no-CUDA fallback path to be exercised
    for real here: constructed, `.is_enabled()` queried. Does not scale
    anything -- there's no real backward pass in this sandbox to scale."""

    def __init__(self, *args, enabled=True, **kwargs):
        self._enabled = enabled

    def is_enabled(self):
        return self._enabled


class _FakeAmpModule:
    GradScaler = _FakeGradScaler


class _FakeAutocast:
    """No-op context manager standing in for torch.autocast -- matches
    what WinCore.precision.AmpContext.autocast() actually needs from it
    on the disabled/CPU path: usable as `with ctx.autocast(): ...` and
    nothing else."""

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeCuda:
    amp = _FakeAmpModule()

    @staticmethod
    def is_available():
        return False


def install():
    mod = types.ModuleType("torch")
    mod.float16 = float16
    mod.float32 = float32
    mod.float64 = float64
    mod.bfloat16 = bfloat16
    mod.float8_e4m3fn = float8_e4m3fn
    mod.float8_e5m2 = float8_e5m2
    mod.cat = cat
    mod.tensor = tensor
    mod.isfinite = isfinite
    mod.randn = randn
    mod.zeros = zeros
    mod.full = full
    mod.ones = ones
    mod.allclose = allclose
    mod.all = all
    mod.cuda = _FakeCuda()
    mod.amp = _FakeAmpModule()
    mod.autocast = _FakeAutocast
    mod.Tensor = FakeTensor
    sys.modules["torch"] = mod
    return mod
