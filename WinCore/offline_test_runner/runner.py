import sys, os, importlib.util, inspect, traceback, tempfile, shutil, pathlib, glob, time

sys.path.insert(0, os.path.dirname(__file__))  # shim pytest.py
import pytest  # noqa: E402

RESULTS = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}
FAILURES = []


class ModuleSkipped(Exception):
    pass


def load_module(path):
    name = "test_mod_" + os.path.basename(path).replace(".py", "")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except pytest.Skipped as e:
        raise ModuleSkipped(str(e))
    return mod


def resolve_fixture_value(fixture_name, module, cache, tmp_base, request_stack):
    if fixture_name in cache:
        return cache[fixture_name]
    if fixture_name == "monkeypatch":
        mp = pytest.MonkeyPatch()
        cache[fixture_name] = mp
        cache.setdefault("__teardowns__", []).append(mp.undo)
        return mp
    if fixture_name == "tmp_path":
        d = tempfile.mkdtemp(dir=tmp_base)
        p = pathlib.Path(d)
        cache[fixture_name] = p
        return p
    obj = getattr(module, fixture_name, None)
    if isinstance(obj, pytest._Fixture):
        sig = inspect.signature(obj.func)
        kwargs = {}
        for pname in sig.parameters:
            kwargs[pname] = resolve_fixture_value(pname, module, cache, tmp_base, request_stack)
        result = obj.func(**kwargs)
        if inspect.isgenerator(result):
            value = next(result)
            cache.setdefault("__teardowns__", []).append(lambda gen=result: _finish_gen(gen))
            cache[fixture_name] = value
            return value
        cache[fixture_name] = result
        return result
    raise LookupError(f"fixture '{fixture_name}' not found")


def _finish_gen(gen):
    try:
        next(gen)
    except StopIteration:
        pass


def get_marks(func):
    return getattr(func, "pytestmark", [])


def run_single(func, module, tmp_base, param_kwargs=None):
    cache = {}
    if param_kwargs:
        cache.update(param_kwargs)
    sig = inspect.signature(func)
    kwargs = {}
    try:
        for pname in sig.parameters:
            if param_kwargs and pname in param_kwargs:
                kwargs[pname] = param_kwargs[pname]
                continue
            kwargs[pname] = resolve_fixture_value(pname, module, cache, tmp_base, [])
        func(**kwargs)
        return ("passed", None)
    except pytest.Skipped as e:
        return ("skipped", str(e))
    except BaseException as e:
        tb = traceback.format_exc()
        return ("failed", tb)
    finally:
        for td in reversed(cache.get("__teardowns__", [])):
            try:
                td()
            except Exception:
                pass


def run_file(path, tmp_base):
    rel = os.path.relpath(path)
    try:
        module = load_module(path)
    except ModuleSkipped as e:
        print(f"SKIP (module) {rel}  ({e})")
        RESULTS["skipped"] += 1
        return
    except Exception as e:
        print(f"[COLLECT ERROR] {rel}: {e}")
        traceback.print_exc()
        RESULTS["errors"] += 1
        return
    test_funcs = [
        (name, obj) for name, obj in vars(module).items()
        if name.startswith("test_") and inspect.isfunction(obj)
    ]
    test_funcs.sort(key=lambda t: inspect.getsourcelines(t[1])[1])
    for name, func in test_funcs:
        marks = get_marks(func)
        skip_reason = None
        parametrize_sets = [None]
        for m in marks:
            if m.name == "skipif":
                cond = m.args[0]
                if cond:
                    skip_reason = m.kwargs.get("reason", "")
            if m.name == "parametrize":
                argnames, argvalues = m.args
                if isinstance(argnames, str):
                    names = [a.strip() for a in argnames.split(",")]
                else:
                    names = list(argnames)
                parametrize_sets = []
                for vals in argvalues:
                    if len(names) == 1:
                        vals = (vals,)
                    parametrize_sets.append(dict(zip(names, vals)))
        if skip_reason is not None:
            print(f"SKIP  {rel}::{name}  ({skip_reason})")
            RESULTS["skipped"] += 1
            continue
        for pset in parametrize_sets:
            label = name if pset is None else f"{name}[{pset}]"
            status, info = run_single(func, module, tmp_base, param_kwargs=pset)
            if status == "passed":
                RESULTS["passed"] += 1
                print(f"PASS  {rel}::{label}")
            elif status == "skipped":
                RESULTS["skipped"] += 1
                print(f"SKIP  {rel}::{label}  ({info})")
            else:
                RESULTS["failed"] += 1
                print(f"FAIL  {rel}::{label}")
                FAILURES.append((f"{rel}::{label}", info))


def main():
    test_dir = sys.argv[1] if len(sys.argv) > 1 else "tests"
    conftest = os.path.join(test_dir, "conftest.py")
    if os.path.exists(conftest):
        load_module(conftest)
    files = sorted(glob.glob(os.path.join(test_dir, "test_*.py")))
    tmp_base = tempfile.mkdtemp(prefix="wincore_test_")
    t0 = time.time()
    for f in files:
        run_file(f, tmp_base)
    dt = time.time() - t0
    shutil.rmtree(tmp_base, ignore_errors=True)
    print("\n" + "=" * 70)
    print(f"passed={RESULTS['passed']} failed={RESULTS['failed']} "
          f"skipped={RESULTS['skipped']} collect_errors={RESULTS['errors']}  "
          f"in {dt:.2f}s")
    if FAILURES:
        print("\n--- FAILURE DETAILS ---")
        for label, tb in FAILURES:
            print(f"\n### {label}\n{tb}")
    return 1 if (RESULTS["failed"] or RESULTS["errors"]) else 0


if __name__ == "__main__":
    sys.exit(main())
