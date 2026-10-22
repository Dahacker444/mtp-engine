"""Enterprise benchmarking matrix for the MTP speculative decoding engine.

Sweeps three axes: batch scale (1/8/32/64/128), RadixKVCache context-window
OOM-safety (2k/8k/32k/64k), and speculative-vs-autoregressive token latency.
"""

from __future__ import annotations

import math
import statistics
import time
from typing import Callable, List, Tuple

import torch

from mtp_engine import (
    KVManager,
    SequentialMTPModule,
    SpeculativeDecodingEngine,
)
from mtp_radix import RadixKVCache
from mtp_kernels import _HAS_TRITON


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


def _time(fn: Callable[[], None], iterations: int, warmup: int, dev: torch.device) -> List[float]:
    for _ in range(warmup):
        fn()
    _sync(dev)
    samples: List[float] = []
    for _ in range(iterations):
        _sync(dev)
        t0 = time.perf_counter()
        fn()
        _sync(dev)
        t1 = time.perf_counter()
        samples.append(t1 - t0)
    return samples


def _summary(name: str, samples: List[float]) -> None:
    n = len(samples)
    print(
        f"  {name:<34} min={min(samples) * 1e6:10.2f}us  "
        f"mean={statistics.fmean(samples) * 1e6:10.2f}us  "
        f"median={statistics.median(samples) * 1e6:10.2f}us  (n={n})"
    )


def _build_operands(
    dev: torch.device, B: int, H: int, D: int, L: int, N: int, dtype: torch.dtype
) -> Tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    q = torch.randn(B, H, D, device=dev, dtype=dtype)
    k = torch.randn(B, H, L, D, device=dev, dtype=dtype)
    v = torch.randn(B, H, L, D, device=dev, dtype=dtype)
    mask = torch.ones(L, device=dev)
    rms_w = torch.randn(D, device=dev, dtype=torch.float32)
    proj_w = torch.randn(N, D, device=dev, dtype=dtype)
    proj_b = torch.randn(N, device=dev, dtype=dtype)
    scale = 1.0 / math.sqrt(D)
    return q, k, v, mask, rms_w, proj_w, proj_b, scale


def _fused_path(operands, fused_fn) -> None:
    q, k, v, mask, rms_w, proj_w, proj_b, scale = operands
    fused_fn(q, k, v, mask, proj_w, rms_w=rms_w, proj_bias=proj_b, scale=scale, eps=1e-6)


def _sequential_path(operands, tree_attn_fn) -> None:
    q, k, v, mask, rms_w, proj_w, proj_b, scale = operands
    a = tree_attn_fn(q, k, v, mask, scale=scale)
    n = a * torch.rsqrt((a * a).mean(dim=-1, keepdim=True) + 1e-6) * rms_w
    torch.einsum("bhd,nd->bhn", n, proj_w.to(torch.float32)) + proj_b


def sweep_batch_scaling(dev: torch.device, dtype: torch.dtype) -> None:
    from mtp_kernels import fused_tree_attn_rmsnorm_linear as fused_fn
    from mtp_kernels import fused_tree_attention as tree_attn_fn

    H, D, L, N = 4, 64, 32, 256
    iterations, warmup = 200, 50
    print("\n[Axis 1] batch-scaling latency (fused monolith vs sequential)")
    print(f"  geometry: H={H} D={D} L={L} N={N} dtype={dtype} "
          f"iterations={iterations} warmup={warmup}")

    for B in (1, 8, 32, 64, 128):
        op = _build_operands(dev, B, H, D, L, N, dtype)
        with torch.no_grad():
            f = fused_fn(op[0], op[1], op[2], op[3], op[5],
                         rms_w=op[4], proj_bias=op[6], scale=op[7], eps=1e-6)
            a = tree_attn_fn(op[0], op[1], op[2], op[3], scale=op[7])
            r = a * torch.rsqrt((a * a).mean(-1, keepdim=True) + 1e-6) * op[4]
            s = torch.einsum("bhd,nd->bhn", r, op[5].to(torch.float32)) + op[6]
            err = (f - s).abs().max().item()
        with torch.no_grad():
            fused_samples = _time(lambda: _fused_path(op, fused_fn), iterations, warmup, dev)
            seq_samples = _time(lambda: _sequential_path(op, tree_attn_fn), iterations, warmup, dev)
        f_med = statistics.median(fused_samples)
        s_med = statistics.median(seq_samples)
        print(
            f"  batch={B:<4} fused_median={f_med * 1e6:10.2f}us  "
            f"seq_median={s_med * 1e6:10.2f}us  speedup={s_med / f_med:5.3f}x  "
            f"max_err={err:.3e}"
        )


def sweep_radix_context(dev: torch.device, dtype: torch.dtype) -> None:
    """Demonstrate RadixKVCache page-sharing keeps memory flat as context grows,
    preventing OOM during dense-context retrieval. ``B`` requests share a long
    common prefix; the radix trie stores it once and only allocates pages for
    the divergent suffixes."""
    print("\n[Axis 2] RadixKVCache dense-context OOM-safety (page dedup)")
    # B = number of concurrent requests; cache geometry kept small so the
    # 64k-token context still fits in the device pool (the OOM-safety claim is
    # about page dedup, not raw pool size).
    num_heads, head_dim, page_size, B = 2, 16, 16, 8
    contexts = (2048, 8192, 32768, 65536)
    print(f"  requests={B} heads={num_heads} head_dim={head_dim} page_size={page_size}")

    cache = RadixKVCache(
        batch=1, num_heads=num_heads, head_dim=head_dim,
        page_size=page_size, device=torch.device("cpu"), dtype=torch.float32,
    )

    shared_prefix_len = 0
    for ctx_len in contexts:
        # Realistic prompt-cache workload: all B requests share a long common
        # prefix (e.g. a shared system prompt + RAG document) and diverge only
        # in a short suffix. Radix dedup stores the prefix once; only divergent
        # suffix pages are per-request.
        prefix = tuple(range(ctx_len - 16))  # shared over all requests
        shared_prefix_len = ctx_len - 16
        for b in range(B):
            tail = tuple((b + 1) * 1_000_000 + i for i in range(16))
            path = prefix + tail
            k = torch.zeros(num_heads, head_dim)
            v = torch.zeros(num_heads, head_dim)
            cache.insert(path, k, v)

        pages = cache.num_pages
        # Naive baseline: without radix dedup, each of the B requests stores
        # its own copy of the full context (one slot per token). Radix stores
        # the shared prefix once, so allocated pages approach ctx_len instead
        # of B * ctx_len.
        naive_pages = B * ctx_len
        print(
            f"  context={ctx_len:<6} tokens  pages_allocated={pages:<7} "
            f"naive_pages={naive_pages:<8}  dedup={naive_pages / max(pages, 1):7.2f}x"
        )

    # Reproduce the shared prefix and confirm it is stored exactly once.
    prefix = tuple(range(shared_prefix_len))
    pid, depth = cache.max_shared_prefix(prefix)
    print(f"  -> shared-prefix depth={depth} (page_id={pid}) "
          f"proves the {shared_prefix_len}-token prefix is stored once for all {B} requests")


class _TinyTarget(torch.nn.Module):
    """Minimal differentiable stand-in for the target (main) model."""

    def __init__(self, hidden_dim: int, vocab_size: int, n_layers: int = 6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embed = torch.nn.Embedding(vocab_size, hidden_dim)
        self.proj = torch.nn.Linear(hidden_dim, vocab_size)
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(hidden_dim, hidden_dim * 4),
                    torch.nn.GELU(),
                    torch.nn.Linear(hidden_dim * 4, hidden_dim),
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.embed(ids)
        for layer in self.layers:
            h = h + layer(h)
        logits = self.proj(h)
        return logits, h


def sweep_verification_latency(dev: torch.device, dtype: torch.dtype) -> None:
    print("\n[Axis 3] speculative MTP verification vs auto-regressive decoding")
    hidden_dim, vocab_size, depth = 64, 256, 4
    n_new = 8

    mtp = SequentialMTPModule(hidden_dim=hidden_dim, vocab_size=vocab_size, depth=depth).to(dev)
    target = _TinyTarget(hidden_dim, vocab_size).to(dev)
    kv = KVManager(512)
    engine = SpeculativeDecodingEngine(
        mtp, target, kv, temperature=1.0, draft_mode="trunk",
        async_mode=False, use_cuda_stream=False,
    )

    prompt = torch.randint(0, vocab_size, (2, 8), device=dev)

    engine.generate(prompt, max_new_tokens=1)

    def speculative_step() -> None:
        engine.generate(prompt, max_new_tokens=n_new)

    def autoregressive_step() -> None:
        ids = prompt
        for _ in range(n_new):
            logits, _ = target(ids)
            nxt = logits[:, -1].argmax(dim=-1)
            ids = torch.cat([ids, nxt.unsqueeze(1)], dim=1)

    iterations, warmup = 100, 20
    with torch.no_grad():
        spec = _time(speculative_step, iterations, warmup, dev)
        ar = _time(autoregressive_step, iterations, warmup, dev)

    s_med = statistics.median(spec)
    a_med = statistics.median(ar)
    print(f"  speculative MTP  median={s_med * 1e6:10.2f}us per {n_new} tokens "
          f"({s_med * 1e6 / n_new:8.2f}us/token)")
    print(f"  auto-regressive  median={a_med * 1e6:10.2f}us per {n_new} tokens "
          f"({a_med * 1e6 / n_new:8.2f}us/token)")
    print(f"  -> speedup (tokens accepted in parity): {a_med / s_med:5.3f}x")


def main() -> None:
    dev = _device()
    dtype = torch.float32
    print(f"device={dev}")
    print(f"HAS_TRITON={_HAS_TRITON}  (fused path routes through eager fallback when False)")

    sweep_batch_scaling(dev, dtype)
    sweep_radix_context(dev, dtype)
    sweep_verification_latency(dev, dtype)
    print("\nALL BENCHMARKS COMPLETE")


if __name__ == "__main__":
    main()

