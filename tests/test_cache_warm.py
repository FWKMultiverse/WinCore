"""
Tests for WinCore.cache.DiskCache.warm() -- concurrent cache
prewarming. Uses real threads (ThreadPoolExecutor) against a real
temp-directory cache, same as test_cache.py's other tests, since this
is exercising real concurrency/locking behavior, not something worth
faking.
"""
import threading
import time

from WinCore.cache import DiskCache


def test_warm_computes_all_missing_keys(tmp_path):
    cache = DiskCache(tmp_path, max_bytes=10 * 1024 * 1024)
    report = cache.warm(range(10), compute_fn=lambda k: f"value-{k}")

    assert report.requested == 10
    assert report.computed == 10
    assert report.already_cached == 0
    assert report.failed == 0
    assert report.errors == []

    for k in range(10):
        # get_or_compute should now hit -- compute_fn here would raise
        # if actually called, proving warm() really stored the value.
        value = cache.get_or_compute(k, lambda: (_ for _ in ()).throw(AssertionError("should be a hit")))
        assert value == f"value-{k}"


def test_warm_skips_already_cached_keys(tmp_path):
    cache = DiskCache(tmp_path, max_bytes=10 * 1024 * 1024)
    cache.get_or_compute(0, lambda: "pre-existing")

    calls = []

    def compute(k):
        calls.append(k)
        return f"value-{k}"

    report = cache.warm(range(3), compute_fn=compute)
    assert report.already_cached == 1
    assert report.computed == 2
    assert 0 not in calls  # never recomputed the pre-existing key


def test_warm_second_call_is_idempotent(tmp_path):
    cache = DiskCache(tmp_path, max_bytes=10 * 1024 * 1024)
    cache.warm(range(5), compute_fn=lambda k: f"v{k}")

    calls = []

    def compute(k):
        calls.append(k)
        return f"v{k}"

    report2 = cache.warm(range(5), compute_fn=compute)
    assert report2.already_cached == 5
    assert report2.computed == 0
    assert calls == []


def test_warm_collects_errors_without_aborting_remaining_keys(tmp_path):
    cache = DiskCache(tmp_path, max_bytes=10 * 1024 * 1024)

    def compute(k):
        if k == 3:
            raise ValueError("boom")
        return f"v{k}"

    report = cache.warm(range(6), compute_fn=compute)
    assert report.computed == 5
    assert report.failed == 1
    assert len(report.errors) == 1
    failed_key, exc = report.errors[0]
    assert failed_key == 3
    assert isinstance(exc, ValueError)

    # the other 5 keys were still stored despite key 3's failure
    for k in range(6):
        if k == 3:
            continue
        value = cache.get_or_compute(k, lambda: (_ for _ in ()).throw(AssertionError("should be a hit")))
        assert value == f"v{k}"


def test_warm_on_error_callback_invoked(tmp_path):
    cache = DiskCache(tmp_path, max_bytes=10 * 1024 * 1024)
    seen = []

    def compute(k):
        if k == 1:
            raise RuntimeError("fail")
        return k

    cache.warm(range(3), compute_fn=compute, on_error=lambda k, exc: seen.append((k, str(exc))))
    assert seen == [(1, "fail")]


def test_warm_empty_keys_returns_zeroed_report(tmp_path):
    cache = DiskCache(tmp_path, max_bytes=10 * 1024 * 1024)
    report = cache.warm([], compute_fn=lambda k: k)
    assert report.requested == 0
    assert report.computed == 0
    assert report.already_cached == 0
    assert report.failed == 0


def test_warm_runs_compute_fn_concurrently(tmp_path):
    """Real concurrency check: N compute_fn calls that each sleep
    briefly should overlap, not run strictly serially, when
    max_workers > 1."""
    cache = DiskCache(tmp_path, max_bytes=10 * 1024 * 1024)

    concurrent_count = {"current": 0, "max_seen": 0}
    lock = threading.Lock()

    def compute(k):
        with lock:
            concurrent_count["current"] += 1
            concurrent_count["max_seen"] = max(concurrent_count["max_seen"], concurrent_count["current"])
        time.sleep(0.05)
        with lock:
            concurrent_count["current"] -= 1
        return k

    cache.warm(range(8), compute_fn=compute, max_workers=4)
    assert concurrent_count["max_seen"] > 1  # actually overlapped, not serial


def test_warm_respects_max_bytes_via_eviction(tmp_path):
    """warm() defers eviction to a single pass at the end (see its own
    docstring) instead of _store()'s normal per-call eviction, but the
    byte budget is still respected by the time warm() returns."""
    cache = DiskCache(tmp_path, max_bytes=500)
    cache.warm(range(30), compute_fn=lambda k: b"x" * 50)
    assert cache.stats.bytes_on_disk <= 500


def test_warm_calls_eviction_exactly_once_not_per_key(tmp_path, monkeypatch):
    """Regression test: an earlier version of warm() routed every key
    through _store() (which calls _evict_if_needed() -- a full
    directory rescan under a cross-process lock -- on every call),
    serializing away the whole point of warming multiple keys
    concurrently. warm() must call _evict_if_needed() exactly ONCE per
    warm() call, regardless of how many keys were written."""
    cache = DiskCache(tmp_path, max_bytes=10 * 1024 * 1024)

    call_count = {"n": 0}
    original = cache._evict_if_needed

    def counting_evict():
        call_count["n"] += 1
        return original()

    monkeypatch.setattr(cache, "_evict_if_needed", counting_evict)

    cache.warm(range(20), compute_fn=lambda k: f"v{k}", max_workers=4)
    assert call_count["n"] == 1


def test_warm_skips_eviction_entirely_when_nothing_to_compute(tmp_path, monkeypatch):
    """If every key is already cached, warm() should not even call
    _evict_if_needed() once -- there's nothing new to evict for."""
    cache = DiskCache(tmp_path, max_bytes=10 * 1024 * 1024)
    cache.warm(range(5), compute_fn=lambda k: f"v{k}")  # pre-warm

    call_count = {"n": 0}
    original = cache._evict_if_needed

    def counting_evict():
        call_count["n"] += 1
        return original()

    monkeypatch.setattr(cache, "_evict_if_needed", counting_evict)

    report = cache.warm(range(5), compute_fn=lambda k: f"v{k}")
    assert report.computed == 0
    assert call_count["n"] == 0


def test_warm_still_evicts_when_all_keys_fail(tmp_path, monkeypatch):
    """Eviction still runs once even if every compute_fn call failed --
    warm() should leave the cache in a consistent state either way,
    not skip the eviction pass just because nothing succeeded."""
    cache = DiskCache(tmp_path, max_bytes=10 * 1024 * 1024)

    call_count = {"n": 0}
    original = cache._evict_if_needed

    def counting_evict():
        call_count["n"] += 1
        return original()

    monkeypatch.setattr(cache, "_evict_if_needed", counting_evict)

    def always_fails(k):
        raise ValueError("boom")

    report = cache.warm(range(5), compute_fn=always_fails)
    assert report.failed == 5
    assert call_count["n"] == 1


def test_store_no_evict_used_by_warm_matches_store_used_by_get_or_compute(tmp_path):
    """_store_no_evict (warm()'s path) and _store (get_or_compute()'s
    path) must produce byte-identical on-disk state for the same
    value -- warm()'s optimization is about WHEN eviction runs, not a
    different write format."""
    cache_a = DiskCache(tmp_path / "a", max_bytes=10 * 1024 * 1024)
    cache_b = DiskCache(tmp_path / "b", max_bytes=10 * 1024 * 1024)

    cache_a.get_or_compute("k", lambda: {"payload": [1, 2, 3]})
    cache_b.warm(["k"], compute_fn=lambda k: {"payload": [1, 2, 3]})

    file_a = next((tmp_path / "a").glob("*.pkl"))
    file_b = next((tmp_path / "b").glob("*.pkl"))
    assert file_a.read_bytes() == file_b.read_bytes()
