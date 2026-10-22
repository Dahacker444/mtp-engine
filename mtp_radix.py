"""Radix-tree KV cache with shared-page deduplication.

This module implements the paged KV cache used during the verification phase of
speculative decoding. The core idea is that concurrent requests sharing a long
common prefix (multi-turn conversations, batch prompt-caching, RAG over a shared
document) store that prefix's KV pages exactly once. Only the divergent suffix
of each request allocates new pages.

Memory model
------------
KV data lives in a single pre-allocated pool of pages. The pool grows
geometrically (x2) so a dense context (64k tokens) does not trigger repeated
reallocations, and the page count is bounded by the number of *distinct*
(prefix, suffix) chunks rather than by the total token count across requests.
Each radix node carries a ``page_id`` and a ``refcount``; no node stores its
full token-path prefix (which would be O(n^2) and defeat dense-context scaling).
Paths are reconstructed only on demand by walking parent links.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch


class RadixNode:
    """A node in the radix trie. Holds a token ``key`` and an optional page id."""

    __slots__ = ("children", "parent", "key", "page_id", "refcount")

    def __init__(self, key: Optional[int] = None, parent: Optional["RadixNode"] = None):
        self.children: Dict[int, "RadixNode"] = {}
        self.parent: Optional["RadixNode"] = parent
        self.key: Optional[int] = key
        self.page_id: Optional[int] = None
        self.refcount: int = 0


class RadixKVCache:
    """Paged KV cache backed by a radix trie with shared-page deduplication.

    Parameters mirror ``PagedKVCache``: ``batch``, ``num_heads``, ``head_dim``,
    ``page_size``, ``device`` and ``dtype``.
    """

    def __init__(
        self,
        batch: int,
        num_heads: int,
        head_dim: int,
        page_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        self.batch = batch
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.device = device
        self.dtype = dtype

        self.root = RadixNode()
        self.num_pages = 0

        # Pool capacity grows geometrically; each page is a dense
        # (batch, num_heads, head_dim) block keyed by page id.
        self._capacity = 0
        self.k_pages = torch.empty((0,), device=device)
        self.v_pages = torch.empty((0,), device=device)

    # ------------------------------------------------------------------ #
    # Page pool management
    # ------------------------------------------------------------------ #
    def _alloc_page(self) -> int:
        pid = self.num_pages
        self.num_pages += 1
        return pid

    def _ensure_page_capacity(self) -> None:
        """Grow the page pool geometrically (x2) instead of by one page per
        token, avoiding repeated reallocations on dense-context insertion."""
        if self.num_pages <= self._capacity:
            return
        new_capacity = max(self._capacity * 2, 64)
        while new_capacity < self.num_pages:
            new_capacity *= 2
        new_k = torch.empty(
            (new_capacity, self.batch, self.num_heads, self.head_dim),
            device=self.device, dtype=self.dtype,
        )
        new_v = torch.empty_like(new_k)
        if self._capacity > 0:
            new_k[: self._capacity].copy_(self.k_pages)
            new_v[: self._capacity].copy_(self.v_pages)
        self.k_pages = new_k
        self.v_pages = new_v
        self._capacity = new_capacity

    # ------------------------------------------------------------------ #
    # Insert / lookup
    # ------------------------------------------------------------------ #
    def insert(self, path: Tuple[int, ...], k: torch.Tensor, v: torch.Tensor) -> int:
        """Insert a token path, sharing pages for already-present prefixes.

        ``k`` / ``v`` have shape ``(num_heads, head_dim)`` (per-token) or
        ``(batch, num_heads, head_dim)`` and represent the KV of the *last*
        token of ``path``. Returns the page id of the terminal node.
        """
        node = self.root
        node.refcount += 1
        for token in path:
            child = node.children.get(token)
            if child is None:
                child = RadixNode(token, node)
                child.page_id = self._alloc_page()
                node.children[token] = child
            node = child
            node.refcount += 1
        self._finalize(node, k, v)
        return node.page_id if node.page_id is not None else -1

    def lookup(self, path: Tuple[int, ...]) -> Tuple[Optional[int], int]:
        """Return ``(page_id, depth)`` for an exact-path lookup.

        ``depth`` is the number of tokens matched; ``page_id`` is the page id of
        the terminal node if the full path exists, else ``None``.
        """
        node = self.root
        depth = 0
        for token in path:
            child = node.children.get(token)
            if child is None:
                return None, depth
            node = child
            depth += 1
        return node.page_id, depth

    def _finalize(self, node: RadixNode, k: torch.Tensor, v: torch.Tensor) -> None:
        """Write the terminal node's KV into its page slot.

        ``k``/``v`` may be shaped ``(num_heads, head_dim)`` (per-token) or
        ``(batch, num_heads, head_dim)``; they are broadcast into the pool slot
        ``(batch, num_heads, head_dim)``.
        """
        self._ensure_page_capacity()
        pid = node.page_id
        kd = k.to(self.dtype)
        vd = v.to(self.dtype)
        if kd.dim() == 2:
            kd = kd.unsqueeze(0)
            vd = vd.unsqueeze(0)
        self.k_pages[pid].copy_(kd)
        self.v_pages[pid].copy_(vd)

    def _path_of(self, node: RadixNode) -> Tuple[int, ...]:
        """Reconstruct a node's token path by walking parent links."""
        path: List[int] = []
        cur: Optional[RadixNode] = node
        while cur is not None and cur.parent is not None:
            path.append(cur.key)
            cur = cur.parent
        return tuple(reversed(path))

    # ------------------------------------------------------------------ #
    # Shared-prefix utilities
    # ------------------------------------------------------------------ #
    def shared_prefix_len(self, a: Tuple[int, ...], b: Tuple[int, ...]) -> int:
        n = 0
        for x, y in zip(a, b):
            if x != y:
                break
            n += 1
        return n

    def max_shared_prefix(self, path: Tuple[int, ...]) -> Tuple[Optional[int], int]:
        """Return ``(page_id, depth)`` of the deepest stored node sharing
        ``path`` as a prefix. Used to find the maximal reusable KV prefix."""
        node = self.root
        best_pid: Optional[int] = None
        depth = 0
        for token in path:
            child = node.children.get(token)
            if child is None:
                break
            node = child
            depth += 1
            if node.page_id is not None:
                best_pid = node.page_id
        return best_pid, depth

    # ------------------------------------------------------------------ #
    # Vectorized / in-kernel batch prefix matching (zero host sync)
    # ------------------------------------------------------------------ #
    def batch_prefix_match(self, ids: torch.Tensor) -> torch.Tensor:
        """Compute the shared-prefix length per request in a single device pass.

        Given a padded token matrix ``ids`` of shape ``(num_requests, max_len)``
        (rows are concurrent requests, columns are token positions), return an
        int32 tensor ``(num_requests,)`` of the longest shared prefix each
        request has against the *canonical* leftmost column path.

        This replaces a Python loop of ``lookup``/``max_shared_prefix`` calls
        (one host round-trip per request) with one tensorized comparison that
        runs entirely on-device (and, on CUDA, through a Triton kernel), so the
        prefix-sharing detection during massive concurrent batches does not
        serialize through the Python interpreter.
        """
        ids = ids.to(device=self.device, dtype=torch.long)
        R, L = ids.shape
        if R <= 1:
            return torch.zeros((R,), dtype=torch.int32, device=self.device)

        # Canonical row for comparison: the leftmost request's path.
        canonical = ids[0:1, :]  # (1, L)

        # Use the Triton kernel ONLY when the input actually lives on CUDA; a
        # CPU tensor cannot be passed to a Triton kernel (it needs a CUDA device
        # pointer). Otherwise fall through to the eager vectorized comparison.
        if ids.is_cuda and _has_triton():
            return _triton_batch_prefix_match(ids, canonical, R, L)

        # Eager fallback: vectorized equality -> first mismatch index.
        diff = (ids != canonical)  # (R, L) bool; True = mismatch at that col
        # Append a "mismatch" sentinel at column L so a fully-matching row
        # resolves to L (no mismatch within the row).
        sentinel = torch.zeros((R, 1), dtype=torch.bool, device=self.device)
        # sentinel is False, so a row with no mismatch gets index L.
        padded = torch.cat([diff, sentinel], dim=1)  # (R, L+1)
        idx = padded.to(torch.int32).argmax(dim=1)  # first True (mismatch) col
        # rows with no mismatch: argmax points at L (the sentinel is False, so
        # argmax over all-False gives 0). Fix: use explicit full-match detection.
        no_mismatch = (~diff).all(dim=1)  # (R,) True where row == canonical fully
        idx = torch.where(no_mismatch, torch.full_like(idx, L), idx)
        return idx.to(torch.int32)


def _has_triton() -> bool:
    try:
        import triton  # noqa: F401
        from mtp_kernels import _HAS_TRITON
        return _HAS_TRITON and torch.cuda.is_available()
    except Exception:
        return False


def _triton_batch_prefix_match(ids: torch.Tensor, canonical: torch.Tensor, R: int, L: int) -> torch.Tensor:
    """Triton kernel: per-row first-mismatch index vs the canonical row."""
    import triton
    import triton.language as tl

    BLOCK: tl.constexpr = 1024

    @triton.jit
    def _prefix_match_kernel(IdsPtr, CanonPtr, OutPtr, R, L, BLOCK_L: tl.constexpr):
        r = tl.program_id(0)
        offs = tl.arange(0, BLOCK_L)
        mask = offs < L
        row = tl.load(IdsPtr + r * L + offs, mask=mask, other=-1)
        can = tl.load(CanonPtr + offs, mask=mask, other=-2)
        eq = (row == can)
        # Find first non-equal position; if all equal, L.
        # Build a shifted comparison: prefix is contiguous equality from left.
        not_eq = ~eq
        # cumulative: find index of first True in not_eq
        idx = tl.min(tl.where(not_eq, offs, L))
        tl.store(OutPtr + r, idx.to(tl.int32))

    out = torch.empty((R,), dtype=torch.int32, device=ids.device)
    BLOCK_L = triton.next_power_of_2(L)
    _prefix_match_kernel[(R,)](ids, canonical, out, R, L, BLOCK_L=BLOCK_L)
    return out


__all__ = ["RadixNode", "RadixKVCache"]

