"""
Generic step-state cache — a KV-cache-shaped tool that is not LLM-
specific.

Why this exists
----------------
"KV cache" in most tooling means one narrow thing: cached
key/value attention tensors for autoregressive LLM decoding. But the
*underlying pattern* — carry forward some per-key tensor state across
sequential steps so you don't recompute it from scratch each step — is
not actually specific to attention. It shows up as:

  - LLM attention: cached key/value tensors, growing one token at a
    time (`mode="append"`).
  - RNN/LSTM: the hidden/cell state carried into the next timestep,
    fully replaced each step, not grown (`mode="replace"`).
  - GNN / GAT / GATv2 / GIN message passing: per-node embeddings from
    the previous propagation round, reused instead of recomputed
    (`mode="replace"`, keyed per layer or per node-batch).
  - PPO / other on-policy RL: rollout buffer state (hidden state for a
    recurrent policy, running advantage stats) carried step to step.
  - MoE: per-expert running token counts or expert-choice state across
    a forward pass.

This module implements the *storage/eviction/compression* mechanics
once, generically, keyed by whatever string key the caller's model
code wants (a layer name, a node-batch id, an expert id, ...) — it has
no attention math, no notion of "token", and no model-family-specific
branches. The model-specific part (what tensor to store, when to call
`update`/`get`) stays in the caller's model code, same as it would
without this module; this only replaces hand-rolled dict-of-tensors
bookkeeping with eviction and optional compression built in.

What this deliberately is NOT
------------------------------
  - Not an attention implementation. It does not concatenate, mask, or
    reshape for you beyond what `mode="append"` does (a plain
    `torch.cat` along `dim`). Attention math is the caller's model
    code, exactly as if it were managing its own dict of tensors.
  - Not automatic — nothing here monkeypatches `nn.Module.forward`.
    You call `.update()` / `.get()` explicitly, same as `WinCore.cache`
    is explicit about `.get_or_compute()`.
  - Not disk-backed (see `WinCore.cache` for that, aimed at dataset
    preprocessing, not live model state). This is in-memory /
    on-device, sized for the lifetime of one generation or one
    training step, not across process restarts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from .precision import Fp8Tensor, dequantize_fp8, quantize_fp8


@dataclass
class _Entry:
    tensor: "object" = None
    fp8: Optional[Fp8Tensor] = None


class StepCache:
    """A keyed store of per-step tensor state, with an optional max
    length (for `mode="append"` growth, e.g. attention KV) and optional
    fp8 compression (for VRAM-bound cases where exact bits matter less
    than fitting more context/state in the same memory budget).

    Args:
        max_len: if set, `update(..., mode="append")` drops the oldest
            entries along `dim` once a key's tensor exceeds this length
            (a sliding window — the standard fixed-context-window KV
            eviction policy). `None` (default) means unbounded growth;
            the caller is responsible for bounding it themselves (e.g.
            by calling `.clear()` between sequences).
        compress: if True, every stored tensor is kept as `Fp8Tensor`
            (via `WinCore.precision.quantize_fp8`) instead of the raw
            dtype, and transparently dequantized on `.get()`. Trades
            some numerical precision for roughly a 2-4x memory
            reduction versus fp16/bf16 storage, and needs the same
            Hopper/Ada+ GPU + torch 2.1+ that `quantize_fp8` needs —
            raises the same clear error at first `.update()` call if
            unavailable, not a silent fallback to uncompressed storage
            (a silent fallback would make the caller's actual memory
            usage a surprise).
    """

    def __init__(self, max_len: Optional[int] = None, compress: bool = False):
        self.max_len = max_len
        self.compress = compress
        self._entries: Dict[str, _Entry] = {}

    def update(self, key: str, tensor, mode: str = "append", dim: int = -2) -> None:
        """Store (or extend) the tensor for `key`.

        Args:
            key: caller-chosen identifier — a layer name, node-batch
                id, expert id, anything hashable-as-string. This module
                attaches no meaning to it.
            tensor: the state to store (e.g. this step's K/V, hidden
                state, node embeddings).
            mode: "append" concatenates onto the existing tensor along
                `dim` (attention-KV-style growth) and applies
                `max_len` eviction if set; "replace" overwrites the
                previous value entirely (RNN/GNN-state-style, where
                only the latest value is meaningful).
            dim: concatenation dimension for `mode="append"`. Ignored
                for "replace". Default `-2`, matching the usual
                `[batch, heads, seq, head_dim]` KV layout where the
                sequence axis is second-to-last — override this for
                other tensor layouts (e.g. `dim=0` for a flat
                `[seq, hidden]` RNN state stack).
        """
        if mode not in ("append", "replace"):
            raise ValueError(f"mode must be 'append' or 'replace', got {mode!r}")

        existing = self._entries.get(key)
        prev = self._decompress(existing) if existing is not None else None

        if mode == "replace" or prev is None:
            new_tensor = tensor
        else:
            import torch

            new_tensor = torch.cat([prev, tensor], dim=dim)
            if self.max_len is not None and new_tensor.shape[dim] > self.max_len:
                new_tensor = new_tensor.narrow(
                    dim, new_tensor.shape[dim] - self.max_len, self.max_len
                )

        self._entries[key] = self._compress(new_tensor)

    def get(self, key: str):
        """Return the current tensor for `key` (dequantized if this
        cache was constructed with `compress=True`), or `None` if
        nothing has been stored for it yet."""
        entry = self._entries.get(key)
        if entry is None:
            return None
        return self._decompress(entry)

    def __contains__(self, key: str) -> bool:
        return key in self._entries

    def keys(self):
        return self._entries.keys()

    def clear(self, key: Optional[str] = None) -> None:
        """Drop one key's state, or all state if `key` is None. Call
        this between independent sequences/episodes/batches so state
        from one doesn't leak (via `mode="append"` growth or a stale
        `mode="replace"` value) into the next."""
        if key is None:
            self._entries.clear()
        else:
            self._entries.pop(key, None)

    def _compress(self, tensor) -> _Entry:
        if not self.compress:
            return _Entry(tensor=tensor)
        return _Entry(fp8=quantize_fp8(tensor))

    def _decompress(self, entry: _Entry):
        if entry.tensor is not None:
            return entry.tensor
        return dequantize_fp8(entry.fp8)
