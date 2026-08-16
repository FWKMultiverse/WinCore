import pytest

torch = pytest.importorskip("torch")

from WinCore.kv import StepCache


def test_get_missing_key_returns_none():
    cache = StepCache()
    assert cache.get("layer0") is None
    assert "layer0" not in cache


def test_append_mode_grows_along_dim():
    """Mirrors attention-KV growth: [batch, heads, seq, head_dim],
    growing along the sequence axis (dim=-2) one step at a time."""
    cache = StepCache()
    step1 = torch.randn(1, 2, 1, 4)
    step2 = torch.randn(1, 2, 1, 4)
    cache.update("layer0", step1, mode="append")
    cache.update("layer0", step2, mode="append")
    result = cache.get("layer0")
    assert result.shape == (1, 2, 2, 4)
    assert torch.allclose(result[:, :, :1], step1)
    assert torch.allclose(result[:, :, 1:], step2)


def test_replace_mode_overwrites_not_grows():
    """Mirrors RNN/GNN state: only the latest value matters."""
    cache = StepCache()
    h1 = torch.randn(1, 8)
    h2 = torch.randn(1, 8)
    cache.update("gru_state", h1, mode="replace")
    cache.update("gru_state", h2, mode="replace")
    result = cache.get("gru_state")
    assert result.shape == (1, 8)
    assert torch.allclose(result, h2)


def test_max_len_evicts_oldest():
    cache = StepCache(max_len=3)
    for i in range(5):
        cache.update("kv", torch.full((1, 1, 1, 1), float(i)), mode="append")
    result = cache.get("kv").flatten().tolist()
    assert result == [2.0, 3.0, 4.0]  # only the last 3 steps survive


def test_multiple_keys_are_independent():
    cache = StepCache()
    cache.update("expert_0", torch.ones(2, 2), mode="replace")
    cache.update("expert_1", torch.zeros(2, 2), mode="replace")
    assert torch.all(cache.get("expert_0") == 1)
    assert torch.all(cache.get("expert_1") == 0)
    assert set(cache.keys()) == {"expert_0", "expert_1"}


def test_clear_single_key():
    cache = StepCache()
    cache.update("a", torch.ones(1), mode="replace")
    cache.update("b", torch.ones(1), mode="replace")
    cache.clear("a")
    assert cache.get("a") is None
    assert cache.get("b") is not None


def test_clear_all():
    cache = StepCache()
    cache.update("a", torch.ones(1), mode="replace")
    cache.clear()
    assert cache.get("a") is None


def test_invalid_mode_raises():
    cache = StepCache()
    with pytest.raises(ValueError):
        cache.update("a", torch.ones(1), mode="sideways")


def test_compress_roundtrip_on_cuda():
    if not (torch.cuda.is_available() and hasattr(torch, "float8_e4m3fn")):
        pytest.skip("needs a CUDA device with float8 dtype support")
    cache = StepCache(compress=True)
    x = torch.randn(4, 4, device="cuda", dtype=torch.float16)
    cache.update("layer0", x, mode="replace")
    restored = cache.get("layer0")
    assert restored.shape == x.shape
    assert restored.dtype == x.dtype
