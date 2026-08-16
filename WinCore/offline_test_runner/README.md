# Offline test runner (no internet, no real pytest)

This sandbox can't `pip install pytest` (no network) and has no torch,
no GPU, no Windows. `pytest.py` here is a small shim implementing just
the pytest surface WinCore's test suite actually uses (`fixture`,
`raises`, `warns`, `skip`, `importorskip`, `mark.parametrize`,
`mark.skipif`, `monkeypatch`, `tmp_path`) so the suite can be executed
for real instead of only read by eye.

Run it from the package root:

    PYTHONPATH="offline_test_runner:$(pwd)" python3 offline_test_runner/runner.py tests

On a normal machine with real pytest installed, ignore this folder
entirely and just run `pytest tests/` as usual — this shim is strictly
a sandbox workaround, not a replacement.
