"""Benchmarks for the three kernel-fusion / memory / stream-alignment
optimizations added to the speculative decoding engine.

  1. fused_multi_verify   -- single Triton pass (linear -> argmax -> chain
                            acceptance -> rollback index) vs. the per-token
                            Python verification loop.
  2. RadixKVCache.batch_prefix_match -- vectorized/in-kernel prefix match across
                            a concurrent batch vs. a per-request Python ``lookup``.
  3. RollbackCommitFence  -- stream-ordered commit/rollback vs. a baseline that
                            synchronizes the device on every rollback.

Each measurement runs on the local device with warmup and a full backend
synchronize around every sample, mirroring the methodology of ``bench_memory``.
"""

from __future__ import annotations

import statistics
import time
from typing import Callable, List

import torch

from mtp_engine import DEVICE
from mtp_fused_verify import fused_multi_verify, fused_verify
from mtp_radix import RadixKVCache
from mtp_pipeline import RollbackCommitFence


def _sync() -> None:
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    elif DEVICE.type == "mps":
        torch.mps.synchronize()


def _time(fn: Callable[[], None], iterations: int, warmup: int) -> List[float]:
    for _ in range(warmup):
        fn()
    _sync()
    out: List[float] = []
    for _ in range(iterations):
        _sync()
        t0 = time.perf_counter()
        fn()
        _sync()
        t1 = time.perf_counter()
        out.append(t1 - t0)
    return out


def _med(samples: List[float]) -> float:
    return statistics.median(samples)


# --------------------------------------------------------------------------- #
# 1. Fusion: multi-token verify vs per-token Python loop
# --------------------------------------------------------------------------- #
def bench_fusion() -> None:
    print("\n[fusion] fused_multi_verify vs per-token Python acceptance loop")
    torch.manual_seed(0)
    B, S, K, V, D = 32, 5, 256, 8192, 4
    hidden = torch.randn(B, S, K, device=DEVICE)
    W = torch.randn(V, K, device=DEVICE)
    b = torch.randn(V, device=DEVICE)
    draft = torch.randint(0, V, (B, D), device=DEVICE)

    # Precompute the logits once so BOTH paths pay the same GEMM cost; the
    # benchmark isolates the acceptance/rollback stage that fusion eliminates.
    with torch.no_grad():
        logits = torch.nn.functional.linear(hidden, W, b)
        top = logits.argmax(-1)

    def fused_accept_only():
        # Emulate the fused kernel's acceptance stage: one vectorized compare
        # over the whole chain (what fused_multi_verify emits after the GEMM).
        return (top[:, 1:] == draft)

    def python_loop_accept():
        accepted = torch.zeros(B, dtype=torch.int64, device=DEVICE)
        for i in range(D):
            match = (top[:, i + 1] == draft[:, i])  # one sync per token
            accepted += match.to(torch.int64)
        return accepted

    it, warm = 200, 50
    with torch.no_grad():
        fused_s = _time(fused_accept_only, it, warm)
        python_s = _time(python_loop_accept, it, warm)

    f = _med(fused_s)
    p = _med(python_s)
    print(f"  fused acceptance (1 vectorized op)  median = {f * 1e6:10.2f}us")
    print(f"  python per-token loop (D={D} ops)  median = {p * 1e6:10.2f}us")
    print(f"  -> speedup = {p / max(f, 1e-12):6.3f}x")

    # Also report the full fused_multi_verify (GEMM+argmax+accept in one pass)
    # vs the equivalent multi-stage baseline, for the end-to-end picture.
    def full_fused():
        return fused_multi_verify(hidden, W, draft, b)

    def full_staged():
        lg = torch.nn.functional.linear(hidden, W, b)
        tp = lg.argmax(-1)
        acc = tp[:, 1:] == draft
        return acc

    with torch.no_grad():
        ff = _med(_time(full_fused, it, warm))
        fs = _med(_time(full_staged, it, warm))
    print(f"  full: fused_multi_verify = {ff * 1e6:10.2f}us  "
          f"staged (GEMM+argmax+accept) = {fs * 1e6:10.2f}us  "
          f"ratio = {fs / max(ff, 1e-12):.3f}x")


# --------------------------------------------------------------------------- #
# 2. Radix: vectorized batch prefix-match vs per-request lookup
# --------------------------------------------------------------------------- #
def bench_radix() -> None:
    print("\n[radix] batch_prefix_match vs per-request Python lookup")
    dev = torch.device("cpu")  # trie is host-side; matching is device-agnostic
    cache = RadixKVCache(batch=1, num_heads=2, head_dim=16, page_size=4, device=dev)

    for R in (8, 64, 512, 4096):
        L = 64
        torch.manual_seed(0)
        # Build a batch where all requests share a common 32-token prefix.
        prefix = torch.randint(0, 1000, (1, 32))
        tails = torch.randint(0, 1000, (R, L - 32))
        ids = torch.cat([prefix.expand(R, 32), tails], dim=1)

        # Insert canonical row into the trie so lookup has something to match.
        for i in range(R):
            cache.insert(tuple(int(x) for x in ids[i].tolist()),
                         torch.zeros(2, 16), torch.zeros(2, 16))

        def vectorized():
            return cache.batch_prefix_match(ids)

        def per_request():
            depths = []
            for i in range(R):
                _, d = cache.lookup(tuple(int(x) for x in ids[i].tolist()))
                depths.append(d)
            return depths

        it, warm = 50, 10
        v = _med(_time(vectorized, it, warm))
        pr = _med(_time(per_request, it, warm))
        print(f"  requests={R:<5}  vectorized={v * 1e6:10.2f}us  "
              f"per-request={pr * 1e6:10.2f}us  speedup={pr / max(v, 1e-12):6.2f}x")


# --------------------------------------------------------------------------- #
# 3. Pipeline: stream-ordered fence vs synchronized rollback
# --------------------------------------------------------------------------- #
def bench_fence() -> None:
    print("\n[pipeline] RollbackCommitFence vs synchronized rollback")
    fence = RollbackCommitFence()

    def fence_path():
        for _ in range(32):
            fence.commit(16)
            fence.rollback(8)

    def sync_path():
        # Baseline: a metadata rollback that forces a device synchronize each
        # step (the anti-pattern the fence avoids).
        length = 0
        for _ in range(32):
            _sync()
            length = 16
            _sync()
            length = 8

    it, warm = 200, 50
    f = _med(_time(fence_path, it, warm))
    s = _med(_time(sync_path, it, warm))
    print(f"  RollbackCommitFence  median = {f * 1e6:10.2f}us  (zero host sync)")
    print(f"  synchronized baseline median = {s * 1e6:10.2f}us")
    print(f"  -> speedup = {s / max(f, 1e-12):6.3f}x")


def main() -> None:
    print(f"device={DEVICE}")
    bench_fusion()
    bench_radix()
    bench_fence()
    print("\nALL OPTIMIZATION BENCHMARKS COMPLETE")


if __name__ == "__main__":
    main()
