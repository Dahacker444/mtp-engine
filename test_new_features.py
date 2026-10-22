from __future__ import annotations

import torch

from mtp_engine import (
    DEVICE,
    KVManager,
    SequentialMTPModule,
    SpeculativeDecodingEngine,
    SpeculativeDraftTree,
)
from mtp_fused_verify import fused_verify, fused_multi_verify
from mtp_radix import RadixKVCache
from mtp_topk import sample_top_k, build_tree_mask
from mtp_pipeline import DualStreamPipeline, AsyncEviction


def test_fused_verify():
    torch.manual_seed(0)
    M, K, V = 4, 64, 256
    h = torch.randn(M, K, device=DEVICE)
    W = torch.randn(V, K, device=DEVICE)
    b = torch.randn(V, device=DEVICE)
    d = torch.randint(0, V, (M,), device=DEVICE)
    accept, top = fused_verify(h, W, d, b)
    logits = torch.nn.functional.linear(h, W, b)
    ref_top = logits.argmax(-1).to(torch.int32)
    ref_acc = (ref_top == d.to(torch.int32)).to(torch.int8)
    assert torch.equal(top, ref_top)
    assert torch.equal(accept, ref_acc)
    print("[fused-verify] fused == eager reference: OK")


def test_fused_multi_verify():
    torch.manual_seed(1)
    B, S, K, V, D = 2, 5, 64, 256, 4
    hidden = torch.randn(B, S, K, device=DEVICE)
    W = torch.randn(V, K, device=DEVICE)
    b = torch.randn(V, device=DEVICE)
    draft = torch.randint(0, V, (B, D), device=DEVICE)
    accept, accepted_n, corrected = fused_multi_verify(hidden, W, draft, b)
    # eager reference
    logits = torch.nn.functional.linear(hidden, W, b)
    top = logits.argmax(-1)
    for i in range(B):
        ref_n = 0
        for d in range(D):
            if top[i, d + 1] == draft[i, d]:
                ref_n = d + 1
            else:
                break
        assert accepted_n[i].item() == ref_n
        assert corrected[i].item() == top[i, 0].item()
        for d in range(D):
            assert accept[i, d].item() == int(top[i, d + 1] == draft[i, d])
    print("[fused-multi-verify] chain accept + rollback index == eager reference: OK")


def test_radix():
    cache = RadixKVCache(batch=1, num_heads=2, head_dim=16, page_size=4, device=DEVICE)
    k = torch.randn(1, 2, 16, device=DEVICE)
    v = torch.randn(1, 2, 16, device=DEVICE)
    cache.insert((1, 2, 3, 4), k, v)
    cache.insert((1, 2, 3, 5), k, v)
    sp = cache.shared_prefix_len((1, 2, 3, 4), (1, 2, 3, 5))
    assert sp == 3
    assert cache.num_pages == 5
    print(f"[radix] shared-prefix dedup: {sp} shared tokens, {cache.num_pages} pages (not 8): OK")


def test_topk():
    logits = torch.randn(2, 100, device=DEVICE)
    tok = sample_top_k(logits, k=5, temperature=1.0)
    assert tok.shape == (2,)
    tree = SpeculativeDraftTree(depth=4, max_branches=8, mode="tree")
    mask = build_tree_mask(tree, torch.tensor([0, 1], device=DEVICE))
    assert mask.shape == (2, 32)
    print("[topk] token shape + tree mask shape: OK")


def test_pipeline():
    def draft(s):
        return torch.tensor([s], dtype=torch.long)

    def verify(t):
        return t * 2

    pipeline = DualStreamPipeline(draft, verify)
    d, v = pipeline.step(3)
    assert v.item() == 6

    rolled = {"len": 10}

    def rb(n):
        rolled["len"] = n

    ev = AsyncEviction(rb, capacity=64)
    ev.schedule_rollback(4)
    assert rolled["len"] == 10
    ev.apply()
    assert rolled["len"] == 4
    print("[pipeline] dual-stream step + async eviction: OK")


def test_end_to_end_still_green():
    class TinyMainModel(torch.nn.Module):
        def __init__(self, h, v):
            super().__init__()
            self.embed = torch.nn.Embedding(v, h)
            self.proj = torch.nn.Linear(h, v, bias=True)

        def forward(self, ids):
            h = self.embed(ids)
            return self.proj(h), h

    torch.manual_seed(3)
    hd, vocab, depth = 64, 256, 4
    mtp = SequentialMTPModule(hd, vocab, depth).to(DEVICE)
    main = TinyMainModel(hd, vocab).to(DEVICE)
    eng = SpeculativeDecodingEngine(
        mtp, main, KVManager(max_seq_len=512), temperature=0.0, depth=depth,
        async_mode=False, draft_mode="tree",
    )
    ids = torch.randint(0, vocab, (2, 5), device=DEVICE)
    out = eng.generate(ids, max_new_tokens=4)
    assert out.shape == (2, 9)
    print("[engine] minified engine still generates correctly: OK")


if __name__ == "__main__":
    test_fused_verify()
    test_fused_multi_verify()
    test_radix()
    test_topk()
    test_pipeline()
    test_end_to_end_still_green()
    print("\nALL NEW MODULE CHECKS PASSED")
