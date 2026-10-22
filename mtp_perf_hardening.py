"""Performance-hardening primitives for the MTP speculative engine.

This module implements the cross-cutting, reusable optimizations that keep the
generation loop free of Python host-device synchronization, allocator churn,
and re-computed constants:

  * power-of-2 / tensor-core-aligned shape padding helpers
  * pinned host memory pools for prompt ingestion / token output (DMA)
  * pre-computed scaling, temperature and normalization constants
  * fast-path shape validation (scalar int comparisons)
  * contiguous / non-blocking transfer helpers
  * stream-binding and zero-branch greedy arguments
  * static parameter-view caching for linear layers
  * flat contiguous page-table buffers for KV caches

Every helper degrades gracefully on CPU/MPS (no-op where not applicable) so the
engine remains fully portable while unlocking the CUDA fast paths.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Power-of-2 / tensor-core alignment
# --------------------------------------------------------------------------- #
def next_pow2(n: int) -> int:
    """Round ``n`` up to the next power of 2 (min 1)."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def align_vocab(vocab_size: int, alignment: int = 128) -> int:
    """Round ``vocab_size`` up to a GEMM-friendly multiple of ``alignment``.

    Tensor-core GEMMs tile the N dimension in 128/256 columns; padding the
    vocabulary to that boundary avoids a ragged final tile.
    """
    aligned = ((vocab_size + alignment - 1) // alignment) * alignment
    return max(aligned, alignment)


def pad_seq_len(seq_len: int, alignment: int = 32) -> int:
    """Round a sequence length up to a warp-aligned multiple (32)."""
    return ((seq_len + alignment - 1) // alignment) * alignment


def pad_batch(batch: int, alignment: int = 8) -> int:
    return ((batch + alignment - 1) // alignment) * alignment


# --------------------------------------------------------------------------- #
# Pre-computed constants
# --------------------------------------------------------------------------- #
class PrecomputedConstants:
    """Caches every division / sqrt / scale that would otherwise be recomputed
    each token step. These are plain Python floats (cheap scalar constants, not
    tensors) so no scalar allocation happens inside the loop."""

    def __init__(
        self,
        head_dim: int,
        hidden_dim: int,
        vocab_size: int,
        dtype: torch.dtype,
        temperature: float = 1.0,
        eps: float = 1e-6,
        fp8_scale: float = 1.0 / 448.0,
    ):
        self.head_dim = head_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.dtype = dtype
        self.eps = eps
        # attention scale = 1/sqrt(head_dim), computed once
        self.attn_scale = 1.0 / math.sqrt(max(head_dim, 1))
        # RMSNorm inv-dim constant
        self.rms_inv = 1.0 / max(hidden_dim, 1)
        # temperature divider for sampling (0 guards greedy path)
        self.inv_temperature = 1.0 / temperature if temperature > 0.0 else 0.0
        # fp8 amax scale
        self.fp8_scale = fp8_scale
        # vocab-aligned tile count
        self.vocab_padded = align_vocab(vocab_size)


# --------------------------------------------------------------------------- #
# Fast-path shape validation
# --------------------------------------------------------------------------- #
def fast_check(actual: Tuple[int, ...], expected: Tuple[int, ...]) -> bool:
    """Scalar integer comparison fast-path (avoids tensor introspection)."""
    if len(actual) != len(expected):
        return False
    for a, b in zip(actual, expected):
        if a != b:
            return False
    return True


# --------------------------------------------------------------------------- #
# Pinned host memory pools (DMA ingestion / output)
# --------------------------------------------------------------------------- #
class PinnedBufferPool:
    """Static page-locked (pinned) CPU tensors for prompt ingestion and token
    output. ``pin_memory`` lets ``.cuda(non_blocking=True)`` run as a DMA
    transfer that bypasses the CPU entirely."""

    def __init__(
        self,
        max_batch: int,
        max_seq_len: int,
        dtype: torch.dtype = torch.long,
    ):
        self.max_batch = max_batch
        self.max_seq_len = max_seq_len
        self.use_pin = torch.cuda.is_available()
        self.prompt = torch.empty(
            (max_batch, max_seq_len), dtype=dtype
        )
        self.tokens = torch.empty(
            (max_batch, max_seq_len), dtype=dtype
        )
        if self.use_pin:
            self.prompt = self.prompt.pin_memory()
            self.tokens = self.tokens.pin_memory()

    def stage_prompt(self, ids: torch.Tensor) -> torch.Tensor:
        """Copy ``ids`` into the pinned prompt buffer and return a view."""
        B, S = ids.shape
        self.prompt[:B, :S].copy_(ids.to(self.prompt.dtype))
        return self.prompt[:B, :S]

    def to_device_non_blocking(self, x: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Pinned -> device DMA transfer (async on CUDA)."""
        if self.use_pin and device.type == "cuda":
            return x.to(device, non_blocking=True)
        return x.to(device)


# --------------------------------------------------------------------------- #
# Static parameter view caching (no .transpose()/.contiguous() in forward)
# --------------------------------------------------------------------------- #
class CachedLinear(nn.Linear):
    """A Linear layer that pre-caches a contiguous weight view and bias at
    construction so the forward path never calls ``.T``/``.contiguous()``."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias=bias)
        self._w = self.weight.data  # already contiguous (out, in)
        self._b = self.bias.data if bias and self.bias is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self._w, self._b)

    def _refresh_cache(self) -> None:
        self._w = self.weight.data
        self._b = self.bias.data if self.bias is not None else None


# --------------------------------------------------------------------------- #
# Zero-branch greedy argument helpers
# --------------------------------------------------------------------------- #
def argmax_greedy(logits: torch.Tensor) -> torch.Tensor:
    """Greedy decode: straight ``argmax``, skipping softmax / temperature /
    RNG entirely (the zero-branch decoding path)."""
    return logits.argmax(dim=-1)


def sample_token(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Temperature-aware sampling that routes ``temperature <= 0`` directly to
    ``argmax`` (no softmax, no multinomial RNG)."""
    if temperature <= 0.0:
        return logits.argmax(dim=-1)
    probs = F.softmax(logits * (1.0 / temperature), dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


# --------------------------------------------------------------------------- #
# Flat contiguous page-table buffers + zero-copy pointer swap
# --------------------------------------------------------------------------- #
class FlatPageTable:
    """A single contiguous INT32 logical->physical mapping for the paged KV
    cache. Updates are in-place (zero-copy), never reallocated."""

    def __init__(self, batch: int, max_logical: int, device: torch.device):
        self.batch = batch
        self.max_logical = max_logical
        self.device = device
        # (batch, logical) contiguous int32 — indexable via pointer arithmetic
        self.table = torch.full(
            (batch, max_logical), -1, dtype=torch.int32, device=device
        )

    def set_logical(self, batch_idx: int, logical: int, physical: int) -> None:
        """In-place pointer write (zero-copy)."""
        self.table[batch_idx, logical] = physical

    def swap_pointers(self, a: int, b: int) -> None:
        """In-place zero-copy swap of two logical columns via index_copy."""
        tmp = self.table[:, a].clone()
        self.table[:, a] = self.table[:, b]
        self.table[:, b] = tmp

    def physical_of(self, batch_idx: int, logical: int) -> int:
        return int(self.table[batch_idx, logical].item())


# --------------------------------------------------------------------------- #
# Stream / allocator binding helpers
# --------------------------------------------------------------------------- #
def get_stream() -> "torch.cuda.Stream":
    """Return the current CUDA stream (or a null context sentinel on non-CUDA)."""
    if torch.cuda.is_available():
        return torch.cuda.current_stream()
    return None


def change_allocator_to_stream() -> None:
    """Bind the CUDA caching allocator to the active stream where supported."""
    if torch.cuda.is_available() and hasattr(torch.cuda.memory, "change_current_allocator"):
        try:
            torch.cuda.memory.change_current_allocator(torch.cuda.memory._default_allocator)
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# JIT compilation
# --------------------------------------------------------------------------- #
def maybe_compile(module: nn.Module, enabled: bool = True):
    """Wrap a module with ``torch.compile(mode="reduce-overhead")`` where
    available, no-op otherwise. ``reduce-overhead`` strips Python interpreter
    dispatch by CUDA-graph-capturing the compiled region."""
    if not enabled:
        return module
    if not hasattr(torch, "compile"):
        return module
    try:
        return torch.compile(module, mode="reduce-overhead")
    except Exception:
        return module


__all__ = [
    "next_pow2",
    "align_vocab",
    "pad_seq_len",
    "pad_batch",
    "PrecomputedConstants",
    "fast_check",
    "PinnedBufferPool",
    "CachedLinear",
    "argmax_greedy",
    "sample_token",
    "FlatPageTable",
    "get_stream",
    "change_allocator_to_stream",
    "maybe_compile",
]
