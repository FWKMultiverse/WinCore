"""Minimal pytest-compatible shim: fixture, raises, warns, skip, importorskip,
mark.parametrize, mark.skipif, MonkeyPatch. Enough surface for WinCore's test suite,
built because real pytest can't be installed in this offline sandbox."""
import functools
import inspect
import os
import sys
import warnings as _warnings_mod

class Skipped(Exception):
    def __init__(self, msg=""):
        super().__init__(msg)
        self.msg = msg

class Failed(AssertionError):
    pass

def skip(msg=""):
    raise Skipped(msg)

def importorskip(name, minversion=None):
    try:
        mod = __import__(name)
        return mod
    except ImportError:
        raise Skipped(f"could not import {name!r}")

class _RaisesContext:
    def __init__(self, expected_exception, match=None):
        self.expected_exception = expected_exception
        self.match = match
        self.value = None
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            raise Failed(f"DID NOT RAISE {self.expected_exception}")
        if not issubclass(exc_type, self.expected_exception):
            return False
        if self.match is not None:
            import re
            if not re.search(self.match, str(exc_val)):
                raise Failed(f"Pattern {self.match!r} not found in {exc_val!r}")
        self.value = exc_val
        return True

def raises(expected_exception, *, match=None):
    return _RaisesContext(expected_exception, match=match)

class _WarnsContext:
    def __init__(self, expected_warning, match=None):
        self.expected_warning = expected_warning
        self.match = match
        self._cm = None
    def __enter__(self):
        self._cm = _warnings_mod.catch_warnings(record=True)
        self._records = self._cm.__enter__()
        _warnings_mod.simplefilter("always")
        return self._records
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cm.__exit__(exc_type, exc_val, exc_tb)
        if exc_type is not None:
            return False
        matched = [w for w in self._records if issubclass(w.category, self.expected_warning)]
        if self.match:
            import re
            matched = [w for w in matched if re.search(self.match, str(w.message))]
        if not matched:
            raise Failed(f"DID NOT WARN {self.expected_warning}")
        return False

def warns(expected_warning, *, match=None):
    return _WarnsContext(expected_warning, match=match)

class _Fixture:
    def __init__(self, func, scope="function", params=None, autouse=False):
        self.func = func
        self.scope = scope
        self.params = params
        self.autouse = autouse
        self.name = func.__name__

def fixture(func=None, *, scope="function", params=None, autouse=False):
    def wrap(f):
        return _Fixture(f, scope=scope, params=params, autouse=autouse)
    if func is not None:
        return wrap(func)
    return wrap

class _MarkDecorator:
    def __init__(self, name, args=None, kwargs=None):
        self.name = name
        self.args = args or ()
        self.kwargs = kwargs or {}
    def __call__(self, func):
        marks = getattr(func, "pytestmark", [])
        marks = marks + [self]
        func.pytestmark = marks
        return func

class _ParametrizeMark:
    def __call__(self, argnames, argvalues, ids=None):
        def deco(func):
            marks = getattr(func, "pytestmark", [])
            marks = marks + [_MarkDecorator("parametrize", (argnames, argvalues), {"ids": ids})]
            func.pytestmark = marks
            return func
        return deco

class _SkipifMark:
    def __call__(self, condition, *, reason=""):
        return _MarkDecorator("skipif", (condition,), {"reason": reason})

class _Mark:
    parametrize = _ParametrizeMark()
    skipif = _SkipifMark()
    def __getattr__(self, item):
        return _MarkDecorator(item)

mark = _Mark()

class MonkeyPatch:
    def __init__(self):
        self._undo = []
    def setattr(self, target, name=None, value=None, raising=True):
        if value is None and name is not None and not isinstance(name, str):
            # setattr(target, value) form with dotted string target
            pass
        if isinstance(target, str) and value is None and name is None:
            raise TypeError("need value")
        if name is None:
            # target is "module.attr" dotted string, value passed positionally as `name`... not used here
            raise TypeError("dotted-string form not supported by shim")
        if raising and not hasattr(target, name):
            raise AttributeError(name)
        had = hasattr(target, name)
        old = getattr(target, name, None)
        self._undo.append(("setattr", target, name, had, old))
        setattr(target, name, value)
    def delattr(self, target, name, raising=True):
        had = hasattr(target, name)
        if not had:
            if raising:
                raise AttributeError(name)
            return
        old = getattr(target, name)
        self._undo.append(("setattr", target, name, had, old))
        delattr(target, name)
    def setenv(self, name, value):
        had = name in os.environ
        old = os.environ.get(name)
        self._undo.append(("env", name, had, old))
        os.environ[name] = value
    def delenv(self, name, raising=True):
        had = name in os.environ
        if not had:
            if raising:
                raise KeyError(name)
            return
        old = os.environ.get(name)
        self._undo.append(("env", name, had, old))
        del os.environ[name]
    def setitem(self, mapping, name, value):
        had = name in mapping
        old = mapping.get(name)
        self._undo.append(("item", mapping, name, had, old))
        mapping[name] = value
    def delitem(self, mapping, name, raising=True):
        had = name in mapping
        if not had:
            if raising:
                raise KeyError(name)
            return
        old = mapping.get(name)
        self._undo.append(("item", mapping, name, had, old))
        del mapping[name]
    def syspath_prepend(self, path):
        sys.path.insert(0, str(path))
        self._undo.append(("syspath", path))
    def chdir(self, path):
        old = os.getcwd()
        self._undo.append(("chdir", old))
        os.chdir(path)
    def context(self):
        return self
    def __enter__(self):
        return self
    def __exit__(self, *a):
        self.undo()
    def undo(self):
        for entry in reversed(self._undo):
            kind = entry[0]
            if kind == "setattr":
                _, target, name, had, old = entry
                if had:
                    setattr(target, name, old)
                else:
                    try:
                        delattr(target, name)
                    except AttributeError:
                        pass
            elif kind == "env":
                _, name, had, old = entry
                if had:
                    os.environ[name] = old
                else:
                    os.environ.pop(name, None)
            elif kind == "item":
                _, mapping, name, had, old = entry
                if had:
                    mapping[name] = old
                else:
                    mapping.pop(name, None)
            elif kind == "syspath":
                _, path = entry
                try:
                    sys.path.remove(str(path))
                except ValueError:
                    pass
            elif kind == "chdir":
                _, old = entry
                os.chdir(old)
        self._undo = []
