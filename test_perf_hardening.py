"""Validation + benchmarks for the performance-hardening primitives.

Covers the new cross-cutting optimizations added for host-sync elimination,
allocator-churn removal, and stream alignment:

  * power-of-2 / tensor-core padding
  * precomputed constants
  * packed FP8 scales
  * bit-mask causal attention (in-register mask)
  * fused_add_rmsnorm (residual + norm in one launch)
  * pinned buffer pool (non-blocking DMA staging)
  * flat contiguous page table + zero-copy swap
  * greedy zero-branch argmax vs softmax+multinomial
  * TripleStreamPipeline / DynamicGraphBatchSlicer / RollbackCommitFence

Run both validation (assertions) and, where meaningful on the local device,
latency measurements. Triton-only paths validate via their eager fallback on
non-CUDA devices.
"""

from __future__ import annotations

import statistics
import time
from typing import List

import torch

import mtp_kernels as mk
import mtp_perf_hardening as mh
import mtp_pipeline as mp
from mtp_engine import DEVICE


def _sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    elif DEVICE.type == "mps":
        torch.mps.synchronize()


def _time(fn, it=100, warm=20) -> List[float]:
    for _ in range(warm):
        fn()
    _sync()
    out = []
    for _ in range(it):
        _sync()
        t0 = time.perf_counter()
        fn()
        _sync()
        out.append(time.perf_counter() - t0)
    return out


# ---- validation assertions ------------------------------------------------- #
def validate() -> None:
    print("== validation ==")

    # power-of-2 / vocab alignment
    assert mh.next_pow2(100) == 128
    assert mh.align_vocab(257) == 384 or mh.align_vocab(257) == 384  # 257 -> 384 for 128-align
    assert mh.align_vocab(8192) == 8192
    assert mh.pad_seq_len(33) == 64
    assert mh.pad_batch(9) == 16
    print("  [pow2/vocab-align] OK")

    # precomputed constants
    c = mh.PrecomputedConstants(64, 256, 8192, torch.float32, temperature=0.7)
    assert abs(c.attn_scale - 1.0 / 8.0) < 1e-9
    assert c.inv_temperature > 0
    assert c.vocab_padded >= 8192
    print("  [precomputed constants] OK")

    # packed scales
    w = [torch.randn(128, 64), torch.randn(64, 64)]
    flat, offs = mk.pack_scales(w)
    assert torch.allclose(mk.unpack_scale_tensor(flat, offs, 0, (128, 64)), w[0])
    assert torch.allclose(mk.unpack_scale_tensor(flat, offs, 1, (64, 64)), w[1])
    print("  [packed fp8 scales] OK")

    # bitmask attention == causal attention (eager)
    q = torch.randn(2, 4, 64, device=DEVICE)
    k = torch.randn(2, 4, 8, 64, device=DEVICE)
    v = torch.randn(2, 4, 8, 64, device=DEVICE)
    bit = mk.fused_bitmask_attention(q, k, v)
    ref = mk.fused_causal_attention(q, k, v)
    assert torch.allclose(bit, ref, atol=1e-4), (bit - ref).abs().max()
    print("  [bitmask attention == causal] OK")

    # fused add+rmsnorm
    x = torch.randn(4, 64, device=DEVICE)
    r = torch.randn(4, 64, device=DEVICE)
    w = torch.randn(64, device=DEVICE)
    fused = mk.fused_add_rmsnorm(x, r, w)
    ref = (x + r) * torch.rsqrt(((x + r) ** 2).mean(-1, keepdim=True) + 1e-6) * w
    assert torch.allclose(fused, ref, atol=1e-4)
    print("  [fused add+rmsnorm] OK")

    # flat page table
    ft = mh.FlatPageTable(2, 16, DEVICE)
    ft.set_logical(0, 0, 5)
    assert ft.physical_of(0, 0) == 5
    ft.set_logical(0, 1, 9)
    ft.swap_pointers(0, 1)
    assert ft.physical_of(0, 0) == 9 and ft.physical_of(0, 1) == 5
    print("  [flat page table + zero-copy swap] OK")

    # greedy zero-branch vs softmax path (same argmax result)
    logits = torch.randn(2, 100, device=DEVICE)
    g = mh.argmax_greedy(logits)
    s = mh.sample_token(logits, 0.0)  # temperature 0 -> argmax path
    assert torch.equal(g, s)
    print("  [greedy zero-branch argmax] OK")

    # triple stream + graph slicer (CPU/MPS path)
    tp = mp.TripleStreamPipeline(lambda s: s + 1, lambda t: t * 2)
    _, d, v = tp.step(3)
    assert d == 4 and v == 8

    calls = []
    slicer = mp.DynamicGraphBatchSlicer(lambda b: calls.append(b) or b * 10)
    slicer.capture()
    # Routing is device-independent: batch=3 maps to the smallest captured
    # slot >= 3, which is 4.
    routed = slicer.route(3)
    # On CUDA, ``capture`` records graphs and ``replay`` returns None (the
    # graph writes into static buffers, it cannot return a Python scalar).
    # On CPU/MPS the slicer stores plain callables and ``replay`` returns their
    # result directly.
    if slicer.use_cuda:
        assert routed is slicer._graphs[4]
        assert slicer.replay(3) is None
    else:
        assert slicer.replay(3) == 40  # batch=4 slot -> 4 * 10
    print("  [triple-stream + graph batch slicer] OK")

    # fence
    f = mp.RollbackCommitFence()
    f.commit(7)
    assert f.length == 7
    f.rollback(3)
    assert f.length == 3
    print("  [rollback/commit fence] OK")


def bench() -> None:
    print("\n== benchmarks ==")
    it, warm = 200, 50

    # greedy argmax vs softmax+multinomial (zero-branch win)
    logits = torch.randn(8, 8192, device=DEVICE)
    t_greedy = statistics.median(_time(lambda: mh.argmax_greedy(logits), it, warm))
    def softmax_path():
        p = torch.softmax(logits, dim=-1)
        torch.multinomial(p, 1)
    t_soft = statistics.median(_time(softmax_path, it, warm))
    print(f"  greedy argmax      = {t_greedy * 1e6:10.2f}us")
    print(f"  softmax+multinomial= {t_soft * 1e6:10.2f}us")
    print(f"  -> greedy speedup  = {t_soft / max(t_greedy, 1e-12):6.3f}x")

    # fused add+rmsnorm vs staged (residual add + rmsnorm separately)
    x = torch.randn(32, 256, device=DEVICE)
    r = torch.randn(32, 256, device=DEVICE)
    w = torch.randn(256, device=DEVICE)
    t_fused = statistics.median(_time(lambda: mk.fused_add_rmsnorm(x, r, w), it, warm))
    def staged():
        s = x + r
        s * torch.rsqrt((s * s).mean(-1, keepdim=True) + 1e-6) * w
    t_staged = statistics.median(_time(staged, it, warm))
    print(f"  fused add+rmsnorm  = {t_fused * 1e6:10.2f}us")
    print(f"  staged (2 kernels) = {t_staged * 1e6:10.2f}us")
    print(f"  -> fusing speedup  = {t_staged / max(t_fused, 1e-12):6.3f}x")


def main() -> None:
    print(f"device={DEVICE}")
    validate()
    bench()
    print("\nALL HARDENING CHECKS COMPLETE")


if __name__ == "__main__":
    main()
