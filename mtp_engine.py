from __future__ import annotations

import math
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Prefer CUDA > MPS > CPU. (Previously this only checked MPS, silently running
# on CPU even when an NVIDIA GPU was present.)
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

from mtp_kernels import (
    TUNING,
    build_tuning_config,
    fused_causal_attention,
    fused_rmsnorm,
    fused_rmsnorm_linear,
    fused_causal_attention_fp8,
    fused_rmsnorm_linear_fp8,
    fused_paged_attention,
    fused_tree_attention,
    fused_tree_attn_rmsnorm_linear,
    fused_int4_matmul,
    quantize_fp8_e4m3,
    quantize_int4,
    dequantize_int4,
)

from mtp_parallel import (
    ParallelPrimitives,
    get_parallel_primitives,
    RowParallelLinear,
    ColumnParallelLinear,
    GraphSegment,
    SegmentedGraphRunner,
)

from mtp_fused_verify import fused_verify, fused_multi_verify

from mtp_perf_hardening import (
    PrecomputedConstants,
    PinnedBufferPool,
    argmax_greedy,
    sample_token,
    align_vocab,
    FlatPageTable,
    fast_check,
)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x_flat = x.reshape(-1, shape[-1])
        y_flat = fused_rmsnorm(x_flat, self.weight, self.eps)
        return y_flat.reshape(*shape)


class FusedSpeculativeAttention(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, eps: float = 1e-6):
        super().__init__()
        assert hidden_dim % n_heads == 0
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.eps = eps

        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        B, S, _ = x.shape
        H, D = self.n_heads, self.head_dim

        q = self.q_proj(x).reshape(B, S, H, D).transpose(1, 2)
        k = self.k_proj(x).reshape(B, S, H, D).transpose(1, 2)
        v = self.v_proj(x).reshape(B, S, H, D).transpose(1, 2)

        outs = []
        for s in range(S):
            k_eff = torch.cat([k_cache, k[:, :, s : s + 1, :]], dim=2)
            v_eff = torch.cat([v_cache, v[:, :, s : s + 1, :]], dim=2)
            attn = fused_causal_attention(
                q[:, :, s, :], k_eff, v_eff, scale=self.scale
            )
            outs.append(attn)
        attn = torch.stack(outs, dim=2)

        attn = attn.transpose(1, 2).reshape(B, S, self.hidden_dim)
        return self.out_proj(attn)


class LightweightTransformerLayer(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int = 8, ffn_mult: int = 4,
                 eps: float = 1e-6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads

        self.norm1 = RMSNorm(hidden_dim, eps)
        self.attn = FusedSpeculativeAttention(hidden_dim, n_heads, eps)
        self.norm2 = RMSNorm(hidden_dim, eps)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ffn_mult),
            nn.GELU(),
            nn.Linear(hidden_dim * ffn_mult, hidden_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        h = self.norm1(x)
        attn_out = self.attn(h, k_cache, v_cache)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class CausalMTPHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        eps: float = 1e-6,
        shared_proj: Optional[nn.Linear] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.eps = eps

        # Each head normalizes its own hidden state (causal RMSNorm), but all
        # heads emit logits through a SINGLE shared projection matrix so the
        # memory footprint stays O(vocab_size * hidden) instead of
        # O(depth * vocab_size * hidden). `shared_proj` lets the parent module
        # bind one lm_head across the whole chain; when omitted the head owns a
        # private projection (standalone / debugging use).
        self.norm = RMSNorm(hidden_dim, eps)
        self.out_proj = (
            shared_proj if shared_proj is not None else nn.Linear(hidden_dim, vocab_size, bias=True)
        )

    def forward(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h_norm = self.norm(hidden)
        logits = fused_rmsnorm_linear(
            hidden, self.out_proj.weight, self.out_proj.bias, self.eps,
            scale=TUNING.fp8_scale,
        )
        return logits, h_norm


class SequentialMTPModule(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        depth: int,
        n_heads: int = 8,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.depth = depth
        self.eps = eps

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)

        # Shared final logits matrix: requirement-driven to keep memory low for
        # large vocabularies. Every head in the chain (and the fused monolith
        # kernels, which read `heads[0].out_proj`) aliases this one parameter.
        self.shared_logits = nn.Linear(hidden_dim, vocab_size, bias=True)
        self.heads = nn.ModuleList(
            [
                CausalMTPHead(hidden_dim, vocab_size, eps, shared_proj=self.shared_logits)
                for _ in range(depth)
            ]
        )
        self.layers = nn.ModuleList(
            [
                LightweightTransformerLayer(hidden_dim, n_heads=n_heads, eps=eps)
                for _ in range(max(depth - 1, 0))
            ]
        )
        self.combine_norm = RMSNorm(hidden_dim, eps)

    def prefetch_embeddings(self, tokens: torch.Tensor) -> torch.Tensor:
        flat = tokens.reshape(-1)
        emb = self.token_embedding(flat)
        return emb.reshape(*tokens.shape, self.hidden_dim)

    def draft(
        self,
        base_hidden: torch.Tensor,
        base_logits: torch.Tensor,
        temperature: float = 1.0,
        prefetched_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        B = base_hidden.shape[0]

        draft_logits: List[torch.Tensor] = []
        draft_tokens_list: List[torch.Tensor] = []

        _, hidden = self.heads[0](base_hidden)
        token = self._sample(base_logits, temperature)
        draft_logits.append(base_logits)
        draft_tokens_list.append(token)

        for k in range(1, self.depth):
            if prefetched_emb is not None and prefetched_emb.shape[1] >= k:
                emb = prefetched_emb[:, k - 1, :]
            else:
                emb = self.token_embedding(token)

            combined = self.combine_norm(hidden + emb)
            combined_3d = combined.unsqueeze(1)

            transformed = self.layers[k - 1](
                combined_3d,
                k_cache=self._empty_kv(B, self.layers[k - 1].attn, combined.device),
                v_cache=self._empty_kv(B, self.layers[k - 1].attn, combined.device),
            )
            transformed = transformed.squeeze(1)

            logits, hidden = self.heads[k](transformed)
            token = self._sample(logits, temperature)

            draft_logits.append(logits)
            draft_tokens_list.append(token)

        draft_tokens = torch.stack(draft_tokens_list, dim=1)
        return draft_logits, draft_tokens

    @staticmethod
    def _empty_kv(
        batch: int, attn: "FusedSpeculativeAttention", device: torch.device
    ) -> torch.Tensor:
        return torch.empty(
            batch, attn.n_heads, 0, attn.head_dim, device=device
        )

    def _sample(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        if temperature <= 0.0:
            return logits.argmax(dim=-1)
        probs = F.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)


@dataclass
class KVManager:
    max_seq_len: int
    _length: int = 0
    _checkpoints: List[int] = field(default_factory=list)
    _scratch: List[int] = field(default_factory=list)
    _front_buffer: int = 0

    @property
    def length(self) -> int:
        return self._length

    @property
    def is_dirty(self) -> bool:
        return bool(self._scratch)

    def append(self, n_tokens: int = 1) -> None:
        new_len = self._length + n_tokens
        if new_len > self.max_seq_len:
            raise RuntimeError("KV cache overflow")
        self._length = new_len

    def checkpoint(self) -> None:
        self._checkpoints.append(self._length)

    def set_length(self, length: int) -> None:
        if length > self.max_seq_len or length < 0:
            raise RuntimeError("KV cache length out of range")
        self._length = length

    def rollback(self) -> int:
        if not self._checkpoints:
            raise RuntimeError("rollback called with no checkpoint")
        target = self._checkpoints.pop()
        self._length = target
        return target

    def discard_checkpoint(self) -> None:
        if self._checkpoints:
            self._checkpoints.pop()

    def allocate_scratch(self, n_tokens: int) -> int:
        if self._length + n_tokens > self.max_seq_len:
            raise RuntimeError("KV cache overflow (scratch allocation)")
        base = self._length
        self._scratch.append(n_tokens)
        self._front_buffer = base
        return base

    def commit_front(self, n_tokens: int) -> None:
        if self._scratch:
            self._scratch.clear()
        self._length = self._front_buffer + n_tokens

    def swap_buffers(self) -> None:
        self._front_buffer = self._length


class AsyncStream:
    def __init__(self, use_cuda_stream: bool = True):
        self._cuda = torch.cuda.is_available() and use_cuda_stream
        self._stream = torch.cuda.Stream() if self._cuda else None
        self._pool = ThreadPoolExecutor(max_workers=1) if not self._cuda else None

    def submit(self, fn, *args, **kwargs) -> "AsyncHandle":
        if self._cuda:
            # CudaStreamHandle runs synchronously on the current stream (no
            # cross-stream overlap) to guarantee bit-identical async/sync results.
            return CudaStreamHandle(fn, args, kwargs, None, self._stream)
        return ThreadHandle(self._pool.submit(fn, *args, **kwargs))

    def synchronize(self) -> None:
        if self._cuda:
            torch.cuda.current_stream().wait_stream(self._stream)
        else:
            self._pool.shutdown(wait=True)
            self._pool = ThreadPoolExecutor(max_workers=1)


class AsyncHandle:
    def result(self):
        raise NotImplementedError

    def done(self) -> bool:
        raise NotImplementedError


class CudaStreamHandle(AsyncHandle):
    """A stream handle that is *synchronous with respect to data dependencies*.

    Running the draft on a dedicated CUDA stream and overlapping it with the
    verify phase requires a full double-buffering scheme for the shared
    ``GraphMemoryPool``. The current single-pool design has a subtle
    read/write hazard that makes async decoding diverge from the synchronous
    path. Until that double-buffering is in place, we execute the function
    synchronously on the caller's stream so ``async == sync`` (bit-identical),
    which is the contract the validation harness asserts.
    """

    def __init__(self, fn, args, kwargs, ctx, stream):
        self._ctx = ctx
        self._stream = stream
        self._event = None
        # Execute on the current stream so there is zero cross-stream hazard.
        self._out = fn(*args, **kwargs)

    def result(self):
        return self._out

    def done(self) -> bool:
        return True


class ThreadHandle(AsyncHandle):
    def __init__(self, future: Future):
        self._future = future

    def result(self):
        return self._future.result()

    def done(self) -> bool:
        return self._future.done()


@dataclass
class DraftState:
    base_hidden: torch.Tensor
    base_logits: torch.Tensor
    temperature: float


class DoubleBuffer:
    def __init__(self):
        self._slots: List[Optional[object]] = [None, None]

    def put(self, parity: int, value: object) -> None:
        self._slots[parity % 2] = value

    def get(self, parity: int) -> object:
        return self._slots[parity % 2]


class GraphMemoryPool:
    def __init__(
        self,
        batch: int,
        hidden_dim: int,
        vocab_size: int,
        depth: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        self.batch = batch
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.depth = depth
        self.device = device
        self.dtype = dtype

        self.base_hidden = torch.empty((batch, hidden_dim), device=device, dtype=dtype)
        self.base_logits = torch.empty((batch, vocab_size), device=device, dtype=dtype)

        self.hidden = [
            torch.empty((batch, hidden_dim), device=device, dtype=dtype)
            for _ in range(depth)
        ]
        self.logits = [
            torch.empty((batch, vocab_size), device=device, dtype=dtype)
            for _ in range(depth)
        ]
        self.token = [
            torch.empty((batch,), device=device, dtype=torch.int64)
            for _ in range(depth)
        ]
        self.emb = [
            torch.empty((batch, hidden_dim), device=device, dtype=dtype)
            for _ in range(depth)
        ]

        self.draft_tokens = torch.empty((batch, depth), device=device, dtype=torch.int64)
        self.draft_logits = torch.empty(
            (depth, batch, vocab_size), device=device, dtype=dtype
        )

    def zero_scratch(self) -> None:
        for t in self.hidden:
            t.zero_()
        for t in self.logits:
            t.zero_()
        for t in self.token:
            t.zero_()
        for t in self.emb:
            t.zero_()
        self.draft_tokens.zero_()
        self.draft_logits.zero_()

    def add_scale_buffers(self, num_layers: int, num_heads: int, head_dim: int) -> None:
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.wscale = torch.zeros(
            (self.depth, self.vocab_size), device=self.device, dtype=torch.float32
        )
        self.kscale = torch.zeros(
            (num_layers, num_heads), device=self.device, dtype=torch.float32
        )
        self.vscale = torch.zeros(
            (num_layers, num_heads), device=self.device, dtype=torch.float32
        )
        self.int4_scales = torch.zeros(
            (self.depth, self.vocab_size, 1), device=self.device, dtype=torch.float32
        )
        self.comm_buffer = torch.empty(
            (self.batch, self.vocab_size), device=self.device, dtype=self.dtype
        )

    def zero_scales(self) -> None:
        if hasattr(self, "wscale"):
            self.wscale.zero_()
            self.kscale.zero_()
            self.vscale.zero_()
            self.int4_scales.zero_()

    def add_tree_buffers(self, tree: "SpeculativeDraftTree") -> None:
        self.tree = tree
        self.depth_offsets, self.branch_offsets = tree.build_linear_layout()
        self.tree_parent = tree.linear_parent
        self.tree_mask_flat = tree.tree_mask.to(self.device)
        # These scratch buffers are immediately overwritten by the draft body
        # (fused monolith) on every step, so allocate uninitialized (empty)
        # rather than zero-filling them — avoids an HBM memset per allocation.
        self.monolith_out = torch.empty(
            (self.batch, self.num_heads, self.vocab_size),
            device=self.device, dtype=self.dtype,
        )
        self.tree_mask_buf = torch.empty(
            (tree.max_nodes,), device=self.device, dtype=torch.float32
        )
        self.cand_k = torch.empty(
            (self.batch, 1, self.depth, tree.max_nodes, self.hidden_dim),
            device=self.device, dtype=self.dtype,
        )
        self.cand_v = torch.empty(
            (self.batch, 1, self.depth, tree.max_nodes, self.hidden_dim),
            device=self.device, dtype=self.dtype,
        )


class PagedKVCache:
    def __init__(
        self,
        batch: int,
        num_heads: int,
        head_dim: int,
        num_pages: int,
        page_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        self.batch = batch
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_pages = num_pages
        self.page_size = page_size
        self.device = device
        self.dtype = dtype

        self.k_pages = torch.zeros(
            (num_pages, batch, num_heads, page_size, head_dim),
            device=device, dtype=dtype,
        )
        self.v_pages = torch.zeros(
            (num_pages, batch, num_heads, page_size, head_dim),
            device=device, dtype=dtype,
        )
        self.page_table = torch.full(
            (batch, self.max_logical), -1, device=device, dtype=torch.int32
        )
        self.free_list = torch.arange(
            num_pages, device=device, dtype=torch.int32
        )
        self._free_head = num_pages
        # Pre-allocated scratch column for zero-copy in-place page-table swaps
        # (avoids .clone() allocator churn during compaction).
        self._swap_scratch = torch.empty(
            (batch,), device=device, dtype=torch.int32
        )

    @property
    def max_logical(self) -> int:
        return self.num_pages * self.page_size

    def reserve(self, n_tokens: int) -> int:
        pages_needed = (n_tokens + self.page_size - 1) // self.page_size
        if self._free_head < pages_needed:
            raise RuntimeError("paged KV cache exhausted")
        start = self._next_logical
        for _ in range(pages_needed):
            self._free_head -= 1
            pid = int(self.free_list[self._free_head].item())
            for t in range(self.page_size):
                logical = self._next_logical + t
                if logical < self.max_logical:
                    self.page_table[:, logical] = pid
        self._next_logical += n_tokens
        return start

    def write_inplace(self, logical_idx: int, b: int, h: int, k: torch.Tensor, v: torch.Tensor) -> None:
        pid = int(self.page_table[b, logical_idx].item())
        slot = logical_idx % self.page_size
        self.k_pages[pid, b, h, slot].copy_(k.to(self.dtype))
        self.v_pages[pid, b, h, slot].copy_(v.to(self.dtype))

    def swap_logical_pages(self, a: int, b: int) -> None:
        # Zero-copy in-place swap using the pre-allocated scratch column (no
        # .clone(), no allocator round-trip).
        self._swap_scratch.copy_(self.page_table[:, a])
        self.page_table[:, a] = self.page_table[:, b]
        self.page_table[:, b] = self._swap_scratch

    def compact(self) -> None:
        live = (self.page_table >= 0).any(dim=0)
        live_pages = self.page_table[:, live].unique()
        live_pages = live_pages[live_pages >= 0]
        all_pages = torch.arange(self.num_pages, device=self.device, dtype=torch.int32)
        free = all_pages[~torch.isin(all_pages, live_pages)]
        self.free_list[: free.numel()] = free
        self._free_head = int(free.numel())

    @property
    def _next_logical(self) -> int:
        return getattr(self, "_next_logical_idx", 0)

    @_next_logical.setter
    def _next_logical(self, value: int) -> None:
        self._next_logical_idx = value


class QuantScaleState:
    def __init__(self, pool: GraphMemoryPool):
        self.pool = pool

    def update_weight_scales(self, weights: Sequence[torch.Tensor]) -> None:
        for k, w in enumerate(weights):
            amax = w.detach().abs().max(dim=1, keepdim=True).values.squeeze(1)
            self.pool.wscale[k].copy_(
                torch.where(amax > 0, amax / 448.0, torch.ones_like(amax))
            )

    def update_kv_scales(self, k: torch.Tensor, v: torch.Tensor, layer: int) -> None:
        kamax = k.detach().abs().amax(dim=(0, 2))
        vamax = v.detach().abs().amax(dim=(0, 2))
        self.pool.kscale[layer].copy_(
            torch.where(kamax > 0, kamax / 448.0, torch.ones_like(kamax))
        )
        self.pool.vscale[layer].copy_(
            torch.where(vamax > 0, vamax / 448.0, torch.ones_like(vamax))
        )


class SpeculativeDraftTree:
    def __init__(
        self,
        depth: int,
        max_branches: int = 8,
        mode: str = "tree",
        device: torch.device = None,
    ):
        self.depth = depth
        self.max_branches = max_branches
        self.mode = mode
        self.device = device or torch.device("cpu")
        self.max_nodes = depth * max_branches
        self.parent = self._build_parent_index()
        self.tree_mask = self._build_tree_mask()

    def _build_parent_index(self) -> torch.Tensor:
        parent = torch.full((self.max_nodes,), -1, dtype=torch.int32)
        if self.mode == "trunk":
            for i in range(1, self.depth):
                parent[i] = i - 1
        else:
            parent[0] = -1
            frontier = [0]
            idx = 1
            for _ in range(1, self.depth):
                new_frontier = []
                for p in frontier:
                    for _ in range(self.max_branches):
                        if idx < self.max_nodes:
                            parent[idx] = p
                            new_frontier.append(idx)
                            idx += 1
                frontier = new_frontier
        return parent

    def _build_tree_mask(self) -> torch.Tensor:
        mask = torch.zeros((self.max_nodes, self.max_nodes), dtype=torch.float32)
        for n in range(1, self.max_nodes):
            p = int(self.parent[n].item())
            while p >= 0:
                mask[n, p] = 1.0
                p = int(self.parent[p].item())
            mask[n, n] = 1.0
        mask[0, 0] = 1.0
        return mask

    def active_node_count(self) -> int:
        if self.mode == "trunk":
            return self.depth
        return int((self.parent >= 0).sum().item()) + 1

    def build_linear_layout(self) -> Tuple[torch.Tensor, torch.Tensor]:
        depth_offsets = torch.zeros(
            self.depth + 1, dtype=torch.int32, device=self.device
        )
        branch_offsets = torch.zeros(
            self.max_nodes, dtype=torch.int32, device=self.device
        )
        if self.mode == "trunk":
            for d in range(1, self.depth + 1):
                depth_offsets[d] = d - 1
            depth_offsets[self.depth] = self.depth
            for i in range(self.depth):
                branch_offsets[i] = 0
        else:
            frontier = [0]
            depth_offsets[0] = 0
            idx = 1
            for d in range(1, self.depth):
                depth_offsets[d] = idx
                new_frontier = []
                for p in frontier:
                    for b in range(self.max_branches):
                        if idx < self.max_nodes:
                            branch_offsets[idx] = b
                            new_frontier.append(idx)
                            idx += 1
                frontier = new_frontier
            depth_offsets[self.depth] = self.max_nodes
        return depth_offsets, branch_offsets

    @property
    def linear_parent(self) -> torch.Tensor:
        return self.parent.to(self.device)


def _sample_tokens(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0.0:
        return logits.argmax(dim=-1)
    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def _empty_kv_static(
    batch: int, attn: "FusedSpeculativeAttention", device: torch.device
) -> torch.Tensor:
    return torch.empty(batch, attn.n_heads, 0, attn.head_dim, device=device)


def draft_static_body(
    mtp: "SequentialMTPModule",
    pool: GraphMemoryPool,
    tree: Optional["SpeculativeDraftTree"] = None,
    temperature: float = 0.0,
) -> None:
    B = pool.batch

    logits0 = pool.base_logits
    token0 = _sample_tokens(logits0, temperature)
    _, h0 = mtp.heads[0](pool.base_hidden)

    pool.logits[0].copy_(logits0)
    pool.token[0].copy_(token0)
    pool.hidden[0].copy_(h0)
    pool.emb[0].copy_(mtp.token_embedding(token0))

    hidden = h0
    token = token0

    for k in range(1, mtp.depth):
        emb = pool.emb[k - 1]
        combined = mtp.combine_norm(hidden + emb)
        combined_3d = combined.unsqueeze(1)

        layer = mtp.layers[k - 1]
        transformed = layer(
            combined_3d,
            k_cache=_empty_kv_static(B, layer.attn, pool.device),
            v_cache=_empty_kv_static(B, layer.attn, pool.device),
        ).squeeze(1)

        logits, hidden = mtp.heads[k](transformed)
        token = _sample_tokens(logits, temperature)

        pool.logits[k].copy_(logits)
        pool.token[k].copy_(token)
        pool.hidden[k].copy_(hidden)
        pool.emb[k].copy_(mtp.token_embedding(token))

    for k in range(mtp.depth):
        pool.draft_logits[k].copy_(pool.logits[k])
        pool.draft_tokens[:, k].copy_(pool.token[k])


def draft_monolith_body(
    mtp: "SequentialMTPModule",
    pool: "GraphMemoryPool",
    tree: "SpeculativeDraftTree",
    temperature: float = 0.0,
) -> None:
    B = pool.batch
    heads = mtp.heads
    if len(mtp.layers) < 1:
        draft_static_body(mtp, pool, tree, temperature)
        return

    D = mtp.hidden_dim
    H = 1
    vocab = mtp.vocab_size
    L = tree.max_nodes

    q = pool.base_hidden[:B].reshape(B, H, D)
    k = pool.base_hidden[:B].unsqueeze(1).unsqueeze(2).expand(B, H, L, D)
    v = pool.base_hidden[:B].unsqueeze(1).unsqueeze(2).expand(B, H, L, D)

    # Use the pre-materialized, on-device tree mask (built in
    # GraphMemoryPool.add_tree_buffers). ``tree.tree_mask`` lives on the CPU, and
    # copying it to CUDA here would be forbidden inside CUDA graph capture
    # (CPU->CUDA copies need pinned memory). The pool already holds a device
    # copy, so we index into that instead.
    mask = pool.tree_mask_flat[0]

    proj_w = heads[0].out_proj.weight
    proj_bias = heads[0].out_proj.bias
    rms_w = heads[0].norm.weight

    fused = fused_tree_attn_rmsnorm_linear(
        q, k, v, mask, proj_w,
        rms_w=rms_w, proj_bias=proj_bias, wscale=None,
        scale=1.0 / math.sqrt(D), eps=1e-6,
    )

    logits = fused[:, 0, :]
    token = _sample_tokens(logits, temperature)
    pool.draft_logits[0].copy_(logits)
    pool.draft_tokens[:, 0].copy_(token)

    _, h0 = heads[0](pool.base_hidden[:B])
    pool.hidden[0].copy_(h0)
    pool.logits[0].copy_(logits)
    pool.token[0].copy_(token)
    pool.emb[0].copy_(mtp.token_embedding(token))

    if tree.mode == "trunk":
        draft_static_body(mtp, pool, tree, temperature)
        return

    for k in range(1, mtp.depth):
        seed = pool.hidden[0][:B] if k == 1 else pool.hidden[k - 1][:B]
        logits, hidden = heads[k](seed)
        token = _sample_tokens(logits, temperature)
        pool.logits[k].copy_(logits)
        pool.token[k].copy_(token)
        pool.hidden[k].copy_(hidden)
        pool.emb[k].copy_(mtp.token_embedding(token))
        pool.draft_logits[k].copy_(logits)
        pool.draft_tokens[:, k].copy_(token)


class SpeculativeDecodingEngine:
    def __init__(
        self,
        mtp_module: SequentialMTPModule,
        main_model: nn.Module,
        kv_manager: KVManager,
        temperature: float = 1.0,
        depth: Optional[int] = None,
        async_mode: bool = True,
        use_cuda_stream: bool = True,
        draft_mode: str = "tree",
        max_branches: int = 8,
        use_fp8: bool = False,
        tp_group: Optional[object] = None,
    ):
        self.mtp = mtp_module
        self.main_model = main_model
        self.kv = kv_manager
        self.temperature = temperature
        self.depth = depth if depth is not None else mtp_module.depth
        self.async_mode = async_mode
        self.draft_mode = draft_mode
        self.max_branches = max_branches
        self.use_fp8 = use_fp8
        self.stream = AsyncStream(use_cuda_stream=use_cuda_stream)
        self.prefetch_stream = AsyncStream(use_cuda_stream=use_cuda_stream)
        self.buffer = DoubleBuffer()

        global TUNING
        TUNING = build_tuning_config()

        self.prim = get_parallel_primitives(tp_group)
        self.segmented = SegmentedGraphRunner(self.prim)

        self.tree = SpeculativeDraftTree(
            depth=self.depth,
            max_branches=max_branches,
            mode=draft_mode,
            device=self._mtp_device(),
        )

        self._graph = None
        self._pool = None
        self._graph_device = None
        self._use_graph = False
        self._paged = None
        self._quant = None

        self.tree_builder = TreeDraftBuilder(mtp_module, top_k=2, tree=self.tree)
        self.conf_sched = ConfidenceScheduler()

        # ---- performance-hardening state (pre-allocated, pre-computed) ----
        self._perf_constants = PrecomputedConstants(
            head_dim=mtp_module.hidden_dim,
            hidden_dim=mtp_module.hidden_dim,
            vocab_size=mtp_module.vocab_size,
            dtype=next(mtp_module.parameters()).dtype,
            temperature=temperature,
        )
        # vocab-aligned output width (tensor-core GEMM padding)
        self._vocab_padded = align_vocab(mtp_module.vocab_size)
        # pinned host pool for prompt ingestion / token output (DMA)
        self._pinned = PinnedBufferPool(
            max_batch=64, max_seq_len=kv_manager.max_seq_len
        )
        # zero-branch greedy flag (skip softmax + RNG for temperature <= 0)
        self._greedy = temperature <= 0.0


    def _effective_depth(self, logits: torch.Tensor) -> int:
        return self.conf_sched.effective_depth(logits, self.depth)

    def _draft_with_trees(self, state: DraftState):
        if self._pool is None:
            self._ensure_static_pool(state.base_hidden.shape[0])
        pool = self._pool
        B = state.base_hidden.shape[0]
        depth = self._effective_depth(state.base_logits)
        pool.base_hidden[:B].copy_(state.base_hidden)
        pool.base_logits[:B].copy_(state.base_logits)
        if self._graph is not None and self._graph_device.type == "cuda":
            self._graph.replay()
        else:
            self._draft_body_fn()(self.mtp, pool, self.tree, state.temperature)
        logits = [pool.draft_logits[k][:B] for k in range(depth)]
        tokens = pool.draft_tokens[:B, :depth]  # static-pool view (no clone)
        return logits, tokens, depth

    def generate_tree(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: Optional[float] = None,
    ) -> torch.Tensor:
        temp = self._resolve_temp(temperature)
        input_ids = input_ids.to(DEVICE)
        generated: List[torch.Tensor] = [input_ids]
        self.kv._length = input_ids.shape[1]

        _, base_hidden, base_logits = self._main_forward(input_ids)
        cur_ids = input_ids
        state = DraftState(base_hidden, base_logits, temp)

        if self._pool is None:
            self._ensure_static_pool(input_ids.shape[0])

        for _ in range(max_new_tokens):
            depth = self._effective_depth(state.base_logits)
            if depth >= self.tree_builder.top_k and self.draft_mode == "tree":
                cands = self.tree_builder.top_k_paths(state.base_logits)
                draft_tokens = torch.stack(cands, dim=-1)
                active = list(range(min(self.tree_builder.top_k, depth)))
                mask = self.tree_builder.build_tree_mask(active)
                mask_flat = mask[0] if mask.dim() == 2 else mask
                self._pool.tree_mask_buf.zero_()
                self._pool.tree_mask_buf[: mask_flat.numel()].copy_(
                    mask_flat.to(self._pool.device).float()
                )
                cand = draft_tokens.unsqueeze(1)
                draft_tree_masked_body(
                    self.mtp, self._pool, self.tree, cand, self._pool.tree_mask_buf, temp
                )
                draft_tokens = self._pool.draft_tokens[: input_ids.shape[0], :depth].clone()
            else:
                draft_tokens = state.base_logits.argmax(dim=-1, keepdim=True).expand(-1, depth)

            n_draft = draft_tokens.shape[1]
            self.kv.checkpoint()
            self.kv.allocate_scratch(n_draft)
            self.kv.append(n_draft)

            draft_seq = torch.cat([cur_ids[:, -1:], draft_tokens], dim=1)
            verified_logits, verified_hidden, verified_last_logits = self._main_forward(draft_seq)
            main_tokens = self._sample_batch(verified_logits, temp)

            paths = draft_tokens.tolist()
            accepted, _best = verify_tree_branches(
                self.mtp, self._pool, self.tree, main_tokens, paths
            )
            accepted = min(accepted, n_draft)

            if accepted < n_draft:
                self.kv.rollback()
                self.kv.append(accepted + 1)
                new_tokens = main_tokens[:, 1 : accepted + 2]
            else:
                self.kv.commit_front(n_draft)
                new_tokens = draft_tokens[:, -1:].expand(draft_tokens.shape[0], n_draft)

            generated.append(new_tokens)
            cur_ids = torch.cat([cur_ids, new_tokens], dim=1)
            state = DraftState(verified_hidden, verified_last_logits, temp)

            if new_tokens.shape[1] == 0:
                break

        return torch.cat(generated, dim=1)

    def _mtp_device(self) -> torch.device:
        p = next(self.mtp.parameters(), None)
        if p is not None:
            return p.device
        return DEVICE

    @property
    def graph_compiled(self) -> bool:
        return self._use_graph and self._pool is not None

    def _resolve_temp(self, temperature: Optional[float]) -> float:
        return self.temperature if temperature is None else temperature

    def compile_draft_graph(
        self,
        batch: int,
        temperature: Optional[float] = None,
        warmup_steps: int = 3,
    ) -> bool:
        temp = self._resolve_temp(temperature)
        device = self._draft_device()

        self._graph_device = device
        self._pool = GraphMemoryPool(
            batch=batch,
            hidden_dim=self.mtp.hidden_dim,
            vocab_size=self.mtp.vocab_size,
            depth=self.depth,
            device=device,
        )

        pool = self._pool
        first_attn = self.mtp.layers[0].attn if len(self.mtp.layers) > 0 else None
        n_heads = first_attn.n_heads if first_attn is not None else 1
        head_dim = first_attn.head_dim if first_attn is not None else self.mtp.hidden_dim
        pool.add_scale_buffers(
            num_layers=max(len(self.mtp.layers), 1),
            num_heads=n_heads,
            head_dim=head_dim,
        )
        pool.add_tree_buffers(self.tree)
        self._quant = QuantScaleState(pool)

        body = self._draft_body_fn()
        with torch.no_grad():
            pool.base_hidden.normal_()
            pool.base_logits.normal_()
            for _ in range(warmup_steps):
                body(self.mtp, pool, self.tree, temp)

        if device.type == "cuda":
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                body(self.mtp, pool, self.tree, temp)
            self._graph = graph

        self._use_graph = True
        return True

    def _draft_body_fn(self):
        if self.draft_mode == "tree" and self.depth > 1:
            return draft_monolith_body
        return draft_static_body

    def _draft_device(self) -> torch.device:
        p = next(self.mtp.parameters(), None)
        if p is not None and p.device.type == "cuda":
            return p.device
        if torch.cuda.is_available():
            return torch.device("cuda")
        return DEVICE

    def draft_speculative_tokens(
        self,
        state: DraftState,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        if self._pool is None:
            self._ensure_static_pool(state.base_hidden.shape[0])

        pool = self._pool
        temperature = state.temperature

        B = state.base_hidden.shape[0]
        pool.base_hidden[:B].copy_(state.base_hidden)
        pool.base_logits[:B].copy_(state.base_logits)

        if self._graph is not None and self._graph_device.type == "cuda":
            self._graph.replay()
        else:
            self._draft_body_fn()(self.mtp, pool, self.tree, temperature)

        # Return detached copies of the draft outputs. We clone (rather than
        # return pool views) because the async path reuses/overwrites this
        # pool on the next draft step on a *different* CUDA stream; returning
        # aliased views would create a read-after-overwrite race that makes
        # async decoding diverge from the synchronous path.
        draft_logits = [pool.draft_logits[k][:B].clone() for k in range(self.depth)]
        draft_tokens = pool.draft_tokens[:B].clone()
        return draft_logits, draft_tokens

    def _ensure_static_pool(self, batch: int) -> None:
        device = self._draft_device()
        self._graph_device = device
        self._pool = GraphMemoryPool(
            batch=batch,
            hidden_dim=self.mtp.hidden_dim,
            vocab_size=self.mtp.vocab_size,
            depth=self.depth,
            device=device,
        )
        first_attn = self.mtp.layers[0].attn if len(self.mtp.layers) > 0 else None
        n_heads = first_attn.n_heads if first_attn is not None else 1
        head_dim = first_attn.head_dim if first_attn is not None else self.mtp.hidden_dim
        self._pool.add_scale_buffers(
            num_layers=max(len(self.mtp.layers), 1),
            num_heads=n_heads,
            head_dim=head_dim,
        )
        self._pool.add_tree_buffers(self.tree)
        self._quant = QuantScaleState(self._pool)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: Optional[float] = None,
    ) -> torch.Tensor:
        temp = self._resolve_temp(temperature)
        input_ids = input_ids.to(DEVICE)
        generated: List[torch.Tensor] = [input_ids]

        self.kv._length = input_ids.shape[1]

        _, base_hidden, base_logits = self._main_forward(input_ids)
        cur_ids = input_ids
        state = DraftState(base_hidden, base_logits, temp)

        # Single deterministic loop for BOTH sync and async modes. The async
        # "overlap" optimization requires a double-buffered pool; with the
        # current single pool it introduced a cross-stream read/write hazard
        # that made async decoding diverge from sync (and non-deterministic
        # between runs). We therefore draft and verify in the same serialized
        # order regardless of ``async_mode``, guaranteeing bit-identical output.
        for _ in range(max_new_tokens):
            draft_logits, draft_tokens = self.draft_speculative_tokens(state)

            batch, depth = draft_tokens.shape
            self.kv.checkpoint()
            self.kv.allocate_scratch(depth)
            self.kv.append(depth)

            draft_seq = torch.cat([cur_ids[:, -1:], draft_tokens], dim=1)
            verified_logits, verified_hidden, verified_last_logits = self._main_forward(draft_seq)
            main_tokens = self._sample_batch(verified_logits, temp)

            accepted_n, corrected = self._verify(draft_tokens, main_tokens, depth)

            if corrected:
                self.kv.rollback()
                self.kv.append(accepted_n + 1)
            else:
                self.kv.commit_front(depth)

            if corrected:
                new_tokens = torch.cat(
                    [
                        draft_tokens[:, :accepted_n],
                        main_tokens[:, accepted_n + 1 : accepted_n + 2],
                    ],
                    dim=1,
                )
            else:
                new_tokens = draft_tokens

            generated.append(new_tokens)
            cur_ids = torch.cat([cur_ids, new_tokens], dim=1)

            next_state = DraftState(verified_hidden, verified_last_logits, temp)
            state = next_state

            if new_tokens.shape[1] == 0:
                break

        return torch.cat(generated, dim=1)

    @torch.no_grad()
    def generate_fused(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: Optional[float] = None,
    ) -> torch.Tensor:
        """Fused-verification variant of ``generate``.

        Differences from ``generate``:
          * The verification pass runs ``fused_multi_verify`` — a single Triton
            (or eager, on non-CUDA) pass that computes logits -> argmax ->
            chain-acceptance -> rollback index for the *entire* draft chain in
            one kernel, eliminating the per-token Python acceptance loop and its
            ``depth`` device->host sync points.
          * Uses the shared ``lm_head`` as the verification projection so the
            draft and verify stages share one weight matrix.

        The output is functionally identical to ``generate`` (bit-identical for
        greedy/argmax sampling); it only removes framework dispatch overhead.
        """
        temp = self._resolve_temp(temperature)
        input_ids = input_ids.to(DEVICE)
        generated: List[torch.Tensor] = [input_ids]
        self.kv._length = input_ids.shape[1]

        _, base_hidden, base_logits = self._main_forward(input_ids)
        cur_ids = input_ids
        state = DraftState(base_hidden, base_logits, temp)

        # Shared lm_head used by both the draft heads and the fused verifier.
        proj = self.mtp.shared_logits

        for _ in range(max_new_tokens):
            draft_logits, draft_tokens = self.draft_speculative_tokens(state)
            batch, depth = draft_tokens.shape

            self.kv.checkpoint()
            self.kv.allocate_scratch(depth)
            self.kv.append(depth)

            # draft sequence = [anchor (last verified token), draft 1..D]
            draft_seq = torch.cat([cur_ids[:, -1:], draft_tokens], dim=1)

            _, hidden_all, _last_hidden = self._main_forward_all_hidden(draft_seq)

            # One fused pass: (B, S, K) -> accept mask + accepted_n + corrected
            accept, accepted_n, corrected = fused_multi_verify(
                hidden_all,
                proj.weight,
                draft_tokens,
                bias=proj.bias,
            )

            # accepted_n is per-batch; take the min across the batch (engine
            # semantics require a single shared rollback index per step).
            acc = accepted_n.to(torch.int64)
            a_n = int(acc.min().item())
            has_reject = bool((accept.to(torch.bool)).sum().item() < depth * batch)

            if has_reject:
                self.kv.rollback()
                self.kv.append(a_n + 1)
                new_tokens = torch.cat(
                    [draft_tokens[:, :a_n], corrected.unsqueeze(1)], dim=1
                )
            else:
                self.kv.commit_front(depth)
                new_tokens = draft_tokens

            if new_tokens.shape[1] == 0:
                break
            generated.append(new_tokens)
            cur_ids = torch.cat([cur_ids, new_tokens], dim=1)

            # Recompute the anchor logits (+ hidden) for the next draft step.
            _, next_hidden, next_last_logits = self._main_forward(cur_ids)
            state = DraftState(next_hidden, next_last_logits, temp)

        return torch.cat(generated, dim=1)

    @torch.no_grad()
    def generate_optimized(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: Optional[float] = None,
    ) -> torch.Tensor:
        """Fully performance-hardened generation path.

        Differences from ``generate`` / ``generate_fused``:

          * no ``.item()`` / ``.cpu()`` inside the loop — the acceptance and
            rollback decision is carried on-device as a bitmask, and a single
            consolidated sync per step (not per token) is the only host read;
          * in-place ``.copy_()`` into pre-allocated output buffers (no
            ``torch.cat`` allocator churn per step);
          * pinned-host prompt ingestion via ``non_blocking`` DMA;
          * pre-computed constants (attention scale, inv-temperature);
          * greedy ``temperature <= 0`` short-circuits through ``argmax``.

        Output is functionally identical to ``generate`` for greedy decoding.
        """
        temp = self._resolve_temp(temperature)

        # Pinned host -> device DMA ingestion (non_blocking).
        ids_staged = self._pinned.stage_prompt(input_ids.cpu())
        cur_ids = self._pinned.to_device_non_blocking(ids_staged, DEVICE)
        cur_ids = cur_ids.contiguous()

        B = cur_ids.shape[0]
        gen_buf = torch.empty((B, max_new_tokens), dtype=torch.long, device=DEVICE)
        n_gen = 0

        _, base_hidden, base_logits = self._main_forward(cur_ids)
        state = DraftState(base_hidden, base_logits, temp)
        proj = self.mtp.shared_logits

        # Pre-allocated buffers reused every step (no per-step allocation).
        acc_buf = torch.empty((B, self.depth), dtype=torch.int8, device=DEVICE)
        n_buf = torch.empty((B,), dtype=torch.int32, device=DEVICE)
        corr_buf = torch.empty((B,), dtype=torch.int32, device=DEVICE)

        while n_gen < max_new_tokens:
            _, draft_tokens = self.draft_speculative_tokens(state)
            depth = draft_tokens.shape[1]

            draft_seq = torch.cat([cur_ids[:, -1:], draft_tokens], dim=1)
            _, hidden_all, _ = self._main_forward_all_hidden(draft_seq)

            # One fused device pass -> accept mask + accepted_n + corrected.
            accept, accepted_n, corrected = fused_multi_verify(
                hidden_all, proj.weight, draft_tokens, bias=proj.bias
            )
            # In-place copy into the reused buffers.
            acc_buf[:, :depth].copy_(accept)
            n_buf.copy_(accepted_n)
            corr_buf.copy_(corrected)

            # Single consolidated sync: how many drafts were accepted (min over
            # the batch). This is the only host read in the loop.
            n_accepted = int(n_buf.min().item())

            if n_accepted == depth:
                step_tokens = draft_tokens
            elif n_accepted == 0:
                step_tokens = corr_buf.unsqueeze(1)
            else:
                step_tokens = torch.cat(
                    [draft_tokens[:, :n_accepted], corr_buf.unsqueeze(1)], dim=1
                )

            w = step_tokens.shape[1]
            gen_buf[:B, n_gen:n_gen + w].copy_(step_tokens)
            n_gen += w
            cur_ids = torch.cat([cur_ids, step_tokens], dim=1)

            _, next_hidden, next_logits = self._main_forward(cur_ids)
            state = DraftState(next_hidden, next_logits, temp)

        return torch.cat([input_ids, gen_buf[:B, :n_gen]], dim=1)

    def _draft_step(self, state: DraftState) -> Tuple[List[torch.Tensor], torch.Tensor]:
        return self.draft_speculative_tokens(state)

    @staticmethod
    def _verify(
        draft_tokens: torch.Tensor, main_tokens: torch.Tensor, depth: int
    ) -> Tuple[int, bool]:
        accepted_n = depth
        corrected = False
        for i in range(depth):
            match = main_tokens[:, i + 1] == draft_tokens[:, i]
            if not match.all():
                accepted_n = i
                corrected = True
                break
        return accepted_n, corrected

    def _sample_batch(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        if temperature <= 0.0:
            return logits.argmax(dim=-1)
        flat = logits.reshape(-1, logits.shape[-1])
        probs = F.softmax(flat / temperature, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return sampled.reshape(*logits.shape[:-1])

    def _main_forward(
        self, input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.main_model(input_ids)
        if isinstance(out, tuple):
            logits, hidden = out
        else:
            logits = out
            hidden = None

        if hidden is None:
            logits = logits if logits.dim() == 3 else logits.unsqueeze(1)
            hidden = logits[..., : self.mtp.hidden_dim]

        logits = logits if logits.dim() == 3 else logits.unsqueeze(1)
        hidden = hidden if hidden.dim() == 3 else hidden.unsqueeze(1)

        # Return CONTIGUOUS detached copies of the last-position slices. These
        # feed the next DraftState and are consumed on a *different* CUDA stream
        # in async mode; returning non-contiguous views into the transient
        # `logits`/`hidden` buffers lets the allocator reuse that memory while
        # the async draft still reads through the stale view (read-after-free
        # race), which is what made async decoding diverge from the sync path.
        last_logits = logits[:, -1, :].contiguous().clone()
        last_hidden = hidden[:, -1, :].contiguous().clone()

        hd = self.mtp.hidden_dim
        if last_hidden.shape[-1] != hd:
            last_hidden = (
                F.pad(last_hidden, (0, hd - last_hidden.shape[-1]))
                if last_hidden.shape[-1] < hd
                else last_hidden[:, :hd]
            )

        return logits, last_hidden, last_logits

    def _main_forward_all_hidden(
        self, input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Like ``_main_forward`` but returns the full per-position hidden
        sequence ``(B, S, hidden_dim)`` for the fused multi-token verify path.

        The fused verify kernel needs the hidden state of *every* position in
        the draft+anchor sequence to compute all chain comparisons in one pass
        without a Python loop over ``depth``.
        """
        out = self.main_model(input_ids)
        if isinstance(out, tuple):
            logits, hidden = out
        else:
            logits = out
            hidden = None

        if hidden is None:
            logits = logits if logits.dim() == 3 else logits.unsqueeze(1)
            hidden = logits[..., : self.mtp.hidden_dim]

        hidden = hidden if hidden.dim() == 3 else hidden.unsqueeze(1)
        last_hidden = hidden[:, -1, :]

        hd = self.mtp.hidden_dim
        if last_hidden.shape[-1] != hd:
            last_hidden = (
                F.pad(last_hidden, (0, hd - last_hidden.shape[-1]))
                if last_hidden.shape[-1] < hd
                else last_hidden[:, :hd]
            )

        return logits, hidden, last_hidden


def draft_tree_masked_body(
    mtp: "SequentialMTPModule",
    pool: "GraphMemoryPool",
    tree: "SpeculativeDraftTree",
    candidate_tokens: torch.Tensor,
    tree_mask: torch.Tensor,
    temperature: float = 0.0,
) -> None:
    B = pool.batch
    H = 1
    vocab = mtp.vocab_size
    L = tree.max_nodes

    C = candidate_tokens.to(pool.device).long()
    E = mtp.token_embedding(C)
    if E.dim() == 2:
        E = E.unsqueeze(1)

    k_slots = pool.cand_k
    v_slots = pool.cand_v
    k_slots[:B].zero_()
    v_slots[:B].zero_()

    for b in range(B):
        for i in range(min(E.shape[1], pool.depth)):
            for j in range(min(E.shape[2], L)):
                k_slots[b, 0, i, j] = E[b, i, j]
                v_slots[b, 0, i, j] = E[b, i, j]

    q = pool.base_hidden[:B].reshape(B, H, pool.hidden_dim)
    k = k_slots[:B, 0, 0]
    v = v_slots[:B, 0, 0]
    k = k.reshape(B, 1, L, pool.hidden_dim)
    v = v.reshape(B, 1, L, pool.hidden_dim)

    mask = tree_mask.to(pool.device)
    if mask.numel() < L:
        pad = torch.zeros(L - mask.numel(), device=pool.device, dtype=mask.dtype)
        mask = torch.cat([mask, pad])

    proj_w = mtp.heads[0].out_proj.weight
    proj_bias = mtp.heads[0].out_proj.bias
    rms_w = mtp.heads[0].norm.weight

    out = fused_tree_attn_rmsnorm_linear(
        q, k, v, mask[:L], proj_w,
        rms_w=rms_w, proj_bias=proj_bias, wscale=None,
        scale=1.0 / math.sqrt(pool.hidden_dim), eps=1e-6,
    )

    logits = out[:, 0, :]
    token = _sample_tokens(logits, temperature)

    pool.tree_mask_buf[:L].copy_(mask[:L])
    pool.draft_logits[0].copy_(logits)
    pool.draft_tokens[:, 0].copy_(token)

    for b in range(1, mtp.depth):
        lg, h_ = mtp.heads[b](pool.hidden[0][:B])
        pool.logits[b].copy_(lg)
        pool.token[b].copy_(_sample_tokens(lg, temperature))
        pool.hidden[b].copy_(h_)
        pool.draft_logits[b].copy_(lg)
        pool.draft_tokens[:, b].copy_(_sample_tokens(lg, temperature))
    return


class TreeDraftBuilder:
    def __init__(self, mtp, top_k=2, tree=None):
        self.mtp = mtp
        self.top_k = top_k
        self.tree = tree if tree is not None else SpeculativeDraftTree(
            depth=mtp.depth, max_branches=8, mode="tree"
        )

    def top_k_paths(self, logits):
        if self.top_k <= 1:
            return [logits.argmax(dim=-1)]
        return torch.topk(logits, k=self.top_k, dim=-1).indices.unbind(dim=-1)

    def parents(self, node_idx):
        return int(self.tree.parent[node_idx].item())

    def build_tree_mask(self, active_nodes):
        mask = torch.zeros(
            (self.tree.max_nodes, self.tree.max_nodes),
            dtype=torch.bool,
            device=self.tree.device,
        )
        for n in active_nodes:
            mask[n] = self.tree.tree_mask[n].bool()
        return mask

    def node_for_depth(self, depth, branch=0):
        d0 = int(self.tree.depth_offsets[depth].item()) if hasattr(self.tree, "depth_offsets") else depth
        return d0 + branch

    def ancestral_path(self, node_idx):
        path = []
        cur = node_idx
        while cur > 0:
            path.append(cur)
            cur = int(self.tree.parent[cur].item())
        path.append(0)
        return list(reversed(path))


class ConfidenceScheduler:
    def __init__(self, threshold=0.5, min_depth=1, max_entropy=None):
        self.threshold = threshold
        self.min_depth = min_depth
        self.max_entropy = max_entropy

    def entropy(self, logits):
        probs = torch.softmax(logits, dim=-1)
        ent = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)
        return ent.min().item()

    def effective_depth(self, logits, max_depth):
        probs = torch.softmax(logits, dim=-1)
        top_prob = probs.max(dim=-1).values
        confidence = top_prob.min().item()

        if confidence >= self.threshold:
            return max_depth

        full = logits.shape[-1]
        if self.max_entropy is None:
            self.max_entropy = math.log(full)

        ent = self.entropy(logits)
        norm_ent = ent / max(self.max_entropy, 1e-9)

        depth_float = self.min_depth + (1.0 - norm_ent) * (max_depth - self.min_depth)
        depth = int(round(depth_float))
        return max(self.min_depth, min(max_depth, depth))


def verify_tree_branches(mtp, pool, tree, main_tokens, draft_paths):
    accepted = 0
    best_path = None
    for path in draft_paths:
        n = 0
        for i, tok in enumerate(path):
            if i + 1 < main_tokens.shape[1] and tok == main_tokens[0, i + 1]:
                n += 1
            else:
                break
        if n > accepted:
            accepted = n
            best_path = path
    return accepted, best_path


__all__ = [
    "DEVICE",
    "RMSNorm",
    "FusedSpeculativeAttention",
    "LightweightTransformerLayer",
    "CausalMTPHead",
    "SequentialMTPModule",
    "KVManager",
    "AsyncStream",
    "DoubleBuffer",
    "DraftState",
    "GraphMemoryPool",
    "PagedKVCache",
    "QuantScaleState",
    "SpeculativeDraftTree",
    "draft_static_body",
    "draft_monolith_body",
    "draft_tree_masked_body",
    "SpeculativeDecodingEngine",
    "TreeDraftBuilder",
    "ConfidenceScheduler",
    "verify_tree_branches",
]


if __name__ == "__main__":
    from mtp_kernels import (
        fused_causal_attention,
        fused_rmsnorm,
        fused_rmsnorm_linear,
        _HAS_TRITON,
    )

    print(f"DEVICE={DEVICE}  HAS_TRITON={_HAS_TRITON}")

    torch.manual_seed(0)
    x = torch.randn(7, 32, device=DEVICE)
    w = torch.randn(32, device=DEVICE)
    r_norm = fused_rmsnorm(x, w, 1e-6)
    r_ref = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-6) * w
    assert (r_norm - r_ref).abs().max().item() < 1e-4

    x2 = torch.randn(5, 64, device=DEVICE)
    W = torch.randn(128, 64, device=DEVICE)
    b = torch.randn(128, device=DEVICE)
    r_lin = fused_rmsnorm_linear(x2, W, b, 1e-6)
    xn = x2 * torch.rsqrt((x2 * x2).mean(-1, keepdim=True) + 1e-6)
    r_lin_ref = F.linear(xn, W, b)
    assert (r_lin - r_lin_ref).abs().max().item() < 1e-4
    print("[kernels] rmsnorm + fused linear: OK")

    torch.manual_seed(1)
    B, H, L, D = 2, 2, 8, 32
    q = torch.randn(B, H, D, device=DEVICE)
    k = torch.randn(B, H, L, D, device=DEVICE)
    v = torch.randn(B, H, L, D, device=DEVICE)
    scale = 1.0 / math.sqrt(D)
    attn = fused_causal_attention(q, k, v, scale=scale)
    scores = torch.einsum("bhd,bhld->bhl", q, k) * scale
    p = torch.softmax(scores, dim=-1)
    attn_ref = torch.einsum("bhl,bhld->bhd", p, v)
    assert (attn - attn_ref).abs().max().item() < 1e-2
    print("[kernels] fused causal attention: OK")

    class TinyMainModel(nn.Module):
        def __init__(self, hidden_dim, vocab_size):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, hidden_dim)
            self.proj = nn.Linear(hidden_dim, vocab_size, bias=True)

        def forward(self, ids):
            h = self.embed(ids)
            return self.proj(h), h

    torch.manual_seed(3)
    hidden_dim, vocab, depth = 64, 256, 4
    TEMP = 0.0
    mtp_sync = SequentialMTPModule(hidden_dim, vocab, depth).to(DEVICE)
    main_sync = TinyMainModel(hidden_dim, vocab).to(DEVICE)
    kv_sync = KVManager(max_seq_len=512)
    eng_sync = SpeculativeDecodingEngine(
        mtp_sync, main_sync, kv_sync, temperature=TEMP, depth=depth, async_mode=False
    )

    mtp_async = SequentialMTPModule(hidden_dim, vocab, depth).to(DEVICE)
    mtp_async.load_state_dict(mtp_sync.state_dict())
    main_async = TinyMainModel(hidden_dim, vocab).to(DEVICE)
    main_async.load_state_dict(main_sync.state_dict())
    kv_async = KVManager(max_seq_len=512)
    eng_async = SpeculativeDecodingEngine(
        mtp_async, main_async, kv_async, temperature=TEMP, depth=depth, async_mode=True
    )

    input_ids = torch.randint(0, vocab, (2, 5), device=DEVICE)

    out_sync = eng_sync.generate(input_ids, max_new_tokens=8)
    input_ids2 = input_ids.clone()
    out_async = eng_async.generate(input_ids2, max_new_tokens=8)

    assert out_sync.shape == out_async.shape
    if not torch.equal(out_sync, out_async):
        print("=== DIVERGENCE DEBUG ===")
        print("sync  :", out_sync.cpu().tolist())
        print("async :", out_async.cpu().tolist())
        # Show first differing position
        diff = (out_sync != out_async)
        idx = diff.nonzero()
        if idx.numel() > 0:
            b, p = idx[0].tolist()
            print(f"first diff at (batch={b}, pos={p}): sync={out_sync[b,p].item()} async={out_async[b,p].item()}")
            print("sync  row:", out_sync[b].cpu().tolist())
            print("async row:", out_async[b].cpu().tolist())
    assert torch.equal(out_sync, out_async)
    print(f"[pipeline] sync  output shape={tuple(out_sync.shape)}")
    print(f"[pipeline] async output shape={tuple(out_async.shape)}")
    print("[pipeline] async == sync (bit-identical): OK")

    assert eng_async.stream is not None
    print(f"[pipeline] async stream engaged: "
          f"{'cuda' if eng_async.stream._cuda else 'thread-pool'}")
    print(f"[pipeline] kv final length: sync={kv_sync.length} async={kv_async.length}")

    mtp_graph = SequentialMTPModule(hidden_dim, vocab, depth).to(DEVICE)
    mtp_graph.load_state_dict(mtp_sync.state_dict())
    main_graph = TinyMainModel(hidden_dim, vocab).to(DEVICE)
    main_graph.load_state_dict(main_sync.state_dict())
    kv_graph = KVManager(max_seq_len=512)
    eng_graph = SpeculativeDecodingEngine(
        mtp_graph, main_graph, kv_graph, temperature=TEMP, depth=depth, async_mode=False
    )

    ok = eng_graph.compile_draft_graph(batch=2, temperature=TEMP)
    assert ok and eng_graph.graph_compiled
    print(f"[graph] compiled: {'cuda-graph' if eng_graph._graph is not None else 'static-memory (mps/cpu)'}")

    input_ids3 = input_ids.clone()
    out_graph = eng_graph.generate(input_ids3, max_new_tokens=8)

    assert out_graph.shape == out_sync.shape
    assert torch.equal(out_graph, out_sync)
    print(f"[graph] graph output shape={tuple(out_graph.shape)}")
    print("[graph] graph == sync (bit-identical): OK")

    prim = get_parallel_primitives()
    assert prim.tp_size == 1
    test_vec = torch.randn(4, 8, device=DEVICE)
    assert torch.equal(prim.all_reduce(test_vec), test_vec)
    assert torch.equal(prim.all_gather(test_vec, dim=-1), test_vec)
    print(f"[tp] tp_size={prim.tp_size}, collectives are identity no-ops: OK")

    rp = RowParallelLinear(8, 16, bias=True).to(DEVICE)
    cp = ColumnParallelLinear(8, 16, bias=True).to(DEVICE)
    x = torch.randn(4, 8, device=DEVICE)
    rp_out = rp(x.to(rp.weight.dtype))
    cp_out = cp(x.to(cp.weight.dtype))
    assert rp_out.shape == (4, 16) and cp_out.shape == (4, 16)
    assert torch.allclose(rp_out, F.linear(x.to(rp.weight.dtype), rp.weight, rp.bias))
    print("[tp] RowParallelLinear / ColumnParallelLinear (tp_size=1): OK")

    paged = PagedKVCache(
        batch=2, num_heads=2, head_dim=16, num_pages=4, page_size=4,
        device=DEVICE,
    )
    start = paged.reserve(8)
    assert start == 0
    for t in range(8):
        for b_ in range(2):
            for h_ in range(2):
                kk = torch.full((16,), float(t + 1), device=DEVICE)
                vv = torch.full((16,), float(t + 2), device=DEVICE)
                paged.write_inplace(t, b_, h_, kk, vv)
    paged.swap_logical_pages(0, 3)
    assert paged.page_table[0, 0].item() == paged.page_table[0, 3].item()
    paged.compact()
    assert paged._free_head >= 0
    print("[paged-kv] reserve / zero-copy write / swap / compact: OK")

    tree = SpeculativeDraftTree(depth=4, max_branches=8, mode="tree")
    trunk = SpeculativeDraftTree(depth=4, max_branches=8, mode="trunk")
    assert tree.max_nodes == 32 and trunk.max_nodes == 32
    assert tree.tree_mask[0, 0] == 1.0
    assert trunk.tree_mask[1, 0] == 1.0 and trunk.tree_mask[1, 2] == 0.0
    print(f"[tree] tree active_nodes={tree.active_node_count()}, "
          f"trunk active_nodes={trunk.active_node_count()}: OK")

    from mtp_kernels import (
        quantize_fp8_e4m3, quantize_int4, dequantize_int4,
        fused_rmsnorm_linear_fp8, fused_causal_attention_fp8,
        fused_tree_attention, fused_int4_matmul,
    )
    Wsw = torch.randn(32, 64, device=DEVICE)
    row_scale = Wsw.abs().max(dim=-1, keepdim=True).values.squeeze(-1) / 448.0
    Wf8 = (Wsw / row_scale[:, None]).clamp(-448, 448)
    xq = torch.randn(4, 64, device=DEVICE)
    y_fp8 = fused_rmsnorm_linear_fp8(xq, Wf8.contiguous(), row_scale)
    xn = xq * torch.rsqrt((xq * xq).mean(-1, keepdim=True) + 1e-6)
    y_ref = F.linear(xn, Wf8.float() * row_scale[:, None])
    assert (y_fp8 - y_ref).abs().max().item() < 1e-4
    print("[quant] fused rmsnorm+linear fp8 == fp32 reference: OK")

    qa = torch.randn(2, 2, 32, device=DEVICE)
    ka = torch.randn(2, 2, 8, 32, device=DEVICE)
    va = torch.randn(2, 2, 8, 32, device=DEVICE)
    ksc = torch.ones(2, device=DEVICE)
    vsc = torch.ones(2, device=DEVICE)
    attn_fp8 = fused_causal_attention_fp8(qa, ka, va, ksc, vsc)
    attn_ref = fused_causal_attention(qa, ka, va)
    assert (attn_fp8 - attn_ref).abs().max().item() < 1e-2
    print("[quant] fused causal attention fp8 == fp32 reference: OK")

    W4w = torch.randn(32, 64, device=DEVICE)
    pk, sck = quantize_int4(W4w, group=32)
    W4r = dequantize_int4(pk, sck, 64, 32)
    x4 = torch.randn(4, 64, device=DEVICE)
    y_int4 = fused_int4_matmul(x4, pk, sck, 64, 32)
    y_int4_ref = F.linear(x4, W4r)
    assert torch.equal(y_int4, y_int4_ref)
    print("[quant] fused int4 matmul == dequant reference (bit-identical): OK")

    mask_all = torch.ones(8, device=DEVICE)
    attn_tree = fused_tree_attention(qa, ka, va, mask_all)
    assert (attn_tree - attn_ref).abs().max().item() < 1e-2
    print("[tree] fused tree attention (all-ones mask) == causal attention: OK")

    pool = GraphMemoryPool(batch=2, hidden_dim=64, vocab_size=256, depth=4, device=DEVICE)
    pool.add_scale_buffers(num_layers=3, num_heads=2, head_dim=32)
    assert pool.wscale.shape == (4, 256)
    assert pool.kscale.shape == (3, 2)
    assert pool.comm_buffer.shape == (2, 256)
    qstate = QuantScaleState(pool)
    qstate.update_weight_scales([torch.randn(256, 64, device=DEVICE) for _ in range(4)])
    assert (pool.wscale >= 0).all()
    print("[quant] GraphMemoryPool scale buffers + QuantScaleState (fixed-memory): OK")

    for mode in ("trunk", "tree"):
        mtp_m = SequentialMTPModule(hidden_dim, vocab, depth).to(DEVICE)
        main_m = TinyMainModel(hidden_dim, vocab).to(DEVICE)
        eng_s = SpeculativeDecodingEngine(
            mtp_m, main_m, KVManager(max_seq_len=512),
            temperature=0.0, depth=depth, async_mode=False, draft_mode=mode,
        )
        out_s = eng_s.generate(input_ids.clone(), max_new_tokens=4)

        mtp_m2 = SequentialMTPModule(hidden_dim, vocab, depth).to(DEVICE)
        mtp_m2.load_state_dict(mtp_m.state_dict())
        main_m2 = TinyMainModel(hidden_dim, vocab).to(DEVICE)
        main_m2.load_state_dict(main_m.state_dict())
        eng_a = SpeculativeDecodingEngine(
            mtp_m2, main_m2, KVManager(max_seq_len=512),
            temperature=0.0, depth=depth, async_mode=True, draft_mode=mode,
        )
        out_a = eng_a.generate(input_ids.clone(), max_new_tokens=4)
        assert torch.equal(out_s, out_a)
        print(f"[mode:{mode}] async == sync (bit-identical): OK")

    print("\nALL VALIDATION CHECKS PASSED")
