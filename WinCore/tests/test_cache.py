from WinCore.cache import DiskCache


def test_miss_then_hit(tmp_path):
    cache = DiskCache(tmp_path, max_bytes=10 * 1024**2)
    calls = []

    def compute():
        calls.append(1)
        return {"value": 42}

    result1 = cache.get_or_compute("key1", compute)
    result2 = cache.get_or_compute("key1", compute)

    assert result1 == {"value": 42}
    assert result2 == {"value": 42}
    assert len(calls) == 1  # only computed once
    assert cache.stats.hits == 1
    assert cache.stats.misses == 1


def test_eviction_respects_byte_budget(tmp_path):
    # Each stored value is a few hundred bytes once pickled; force a tiny
    # budget so eviction kicks in almost immediately.
    cache = DiskCache(tmp_path, max_bytes=600)

    for i in range(20):
        cache.get_or_compute(f"key{i}", lambda i=i: b"x" * 200)

    assert cache.stats.bytes_on_disk <= 600
    assert cache.stats.evictions > 0
    assert len(cache) < 20


def test_lru_order_keeps_recently_used_entry(tmp_path):
    cache = DiskCache(tmp_path, max_bytes=900)

    cache.get_or_compute("a", lambda: b"x" * 200)
    cache.get_or_compute("b", lambda: b"x" * 200)
    cache.get_or_compute("c", lambda: b"x" * 200)
    # touch "a" again so it's no longer the least-recently-used entry
    cache.get_or_compute("a", lambda: b"x" * 200)
    # this push should evict "b" (now least recently used), not "a"
    cache.get_or_compute("d", lambda: b"x" * 200)
    cache.get_or_compute("e", lambda: b"x" * 200)

    calls = []
    cache.get_or_compute("a", lambda: calls.append(1) or b"x" * 200)
    assert calls == []  # "a" was still cached, not recomputed


def test_persists_across_new_instance_pointed_at_same_directory(tmp_path):
    cache1 = DiskCache(tmp_path, max_bytes=10 * 1024**2)
    cache1.get_or_compute("persisted", lambda: "value")

    cache2 = DiskCache(tmp_path, max_bytes=10 * 1024**2)
    calls = []
    result = cache2.get_or_compute("persisted", lambda: calls.append(1) or "value")

    assert result == "value"
    assert calls == []  # found on disk from cache1, not recomputed


def test_clear_removes_all_entries(tmp_path):
    cache = DiskCache(tmp_path, max_bytes=10 * 1024**2)
    cache.get_or_compute("a", lambda: 1)
    cache.get_or_compute("b", lambda: 2)

    cache.clear()

    assert len(cache) == 0
    assert cache.stats.bytes_on_disk == 0


def test_no_max_bytes_auto_sizes_from_real_free_disk_space(tmp_path, monkeypatch):
    import shutil

    real_usage = shutil.disk_usage(tmp_path)
    # Snapshot one real disk_usage() reading and pin shutil.disk_usage
    # to return it for the rest of this test, instead of letting the
    # test's own call and DiskCache's internal call be two separate
    # real syscalls a few milliseconds apart. On a live machine
    # (background writes, Search indexing, browser cache, etc.) free
    # space can drift between those two calls -- confirmed on a real
    # Windows box: this test failed once (off by ~68KB out of ~64GB
    # free) then passed on an immediate retry with no code change in
    # between. Not a DiskCache bug -- a test race against a moving
    # target. Pinning to one snapshot keeps the assertion exact while
    # still exercising a genuine, real free-space number.
    monkeypatch.setattr(shutil, "disk_usage", lambda path: real_usage)

    cache = DiskCache(tmp_path)  # no max_bytes -- should auto-size
    assert cache.max_bytes == int(real_usage.free * 0.5)


def test_free_space_fraction_is_respected(tmp_path, monkeypatch):
    import shutil

    real_usage = shutil.disk_usage(tmp_path)
    # Same race as above -- pin the reading so both calls agree.
    monkeypatch.setattr(shutil, "disk_usage", lambda path: real_usage)

    cache = DiskCache(tmp_path, free_space_fraction=0.1)
    assert cache.max_bytes == int(real_usage.free * 0.1)


def test_explicit_max_bytes_always_overrides_auto_sizing(tmp_path):
    cache = DiskCache(tmp_path, max_bytes=777)
    assert cache.max_bytes == 777


def test_shared_budget_enforced_across_independent_instances(tmp_path):
    # Simulates DataLoader(num_workers=2) on Windows: two independent
    # DiskCache instances (spawn, not fork -- no shared memory) pointed
    # at the same directory. Each writes under a budget that ONLY it
    # could see would be respected -- the real bug this guards against
    # is the *combined* directory blowing past max_bytes because
    # neither worker's local bookkeeping knew about the other's files.
    max_bytes = 1200
    worker_a = DiskCache(tmp_path, max_bytes=max_bytes)
    worker_b = DiskCache(tmp_path, max_bytes=max_bytes)

    for i in range(10):
        worker_a.get_or_compute(f"a{i}", lambda: b"x" * 200)
        worker_b.get_or_compute(f"b{i}", lambda: b"x" * 200)

    actual_bytes_on_disk = sum(p.stat().st_size for p in tmp_path.glob("*.pkl"))
    assert actual_bytes_on_disk <= max_bytes


def test_hit_from_sibling_instance_recognized_without_recompute(tmp_path):
    # worker_b is constructed FIRST (empty directory, empty local
    # _index), then worker_a writes the key afterward. worker_b's
    # local _index never learns about it via its own _store() or
    # construction-time scan -- the only way worker_b can see this as
    # a hit is by checking the filesystem directly, not local state.
    worker_b = DiskCache(tmp_path, max_bytes=10 * 1024**2)
    worker_a = DiskCache(tmp_path, max_bytes=10 * 1024**2)
    worker_a.get_or_compute("shared", lambda: "value")

    calls = []
    result = worker_b.get_or_compute("shared", lambda: calls.append(1) or "value")

    assert result == "value"
    assert calls == []
    assert worker_b.stats.hits == 1


def test_auto_sizing_falls_back_safely_if_disk_usage_fails(tmp_path, monkeypatch):
    import shutil
    import WinCore.cache as cache_mod

    def boom(path):
        raise OSError("simulated failure (e.g. unsupported network drive)")

    monkeypatch.setattr(shutil, "disk_usage", boom)
    cache = DiskCache(tmp_path)
    assert cache.max_bytes == cache_mod._FALLBACK_MAX_BYTES
