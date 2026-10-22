from __future__ import annotations

import torch
import torch.nn as nn

from mtp_engine import (
    DEVICE,
    KVManager,
    SequentialMTPModule,
    SpeculativeDecodingEngine,
    TreeDraftBuilder,
    ConfidenceScheduler,
    verify_tree_branches,
)
from mtp_kernels import fused_tree_attention


class TinyMainModel(nn.Module):
    def __init__(self, h, v):
        super().__init__()
        self.embed = nn.Embedding(v, h)
        self.proj = nn.Linear(h, v, bias=True)

    def forward(self, ids):
        h = self.embed(ids)
        return self.proj(h), h


def test_adaptive_depth_wired():
    torch.manual_seed(0)
    mtp = SequentialMTPModule(64, 256, 4).to(DEVICE)
    main = TinyMainModel(64, 256).to(DEVICE)
    eng = SpeculativeDecodingEngine(mtp, main, KVManager(512), temperature=0.0,
                                    depth=4, draft_mode="tree")
    assert eng.conf_sched is not None
    assert eng.tree_builder is not None
    high = torch.tensor([[5.0, 0.0, 0.0]], device=DEVICE)
    low = torch.tensor([[0.0, 0.0, 0.0]], device=DEVICE)
    assert eng._effective_depth(high) == 4
    assert eng._effective_depth(low) == 1
    print("[tree-cycle] ConfidenceScheduler wired into engine._effective_depth: OK")


def test_tree_generation_end_to_end():
    torch.manual_seed(3)
    mtp = SequentialMTPModule(64, 256, 4).to(DEVICE)
    main = TinyMainModel(64, 256).to(DEVICE)
    for mode in ("tree", "trunk"):
        eng = SpeculativeDecodingEngine(mtp, main, KVManager(512), temperature=0.0,
                                        depth=4, draft_mode=mode)
        ids = torch.randint(0, 256, (2, 5), device=DEVICE)
        out = eng.generate_tree(ids, max_new_tokens=4)
        assert out.shape == (2, 9), (mode, out.shape)
        assert torch.equal(out[:, :5], ids)
        print(f"[tree-cycle] generate_tree '{mode}' mode: shape={tuple(out.shape)}, prompt preserved: OK")


def test_tree_mask_coupled_into_attention():
    torch.manual_seed(7)
    mtp = SequentialMTPModule(64, 256, 4).to(DEVICE)
    main = TinyMainModel(64, 256).to(DEVICE)
    eng = SpeculativeDecodingEngine(mtp, main, KVManager(512), temperature=0.0,
                                    depth=4, draft_mode="tree")
    ids = torch.randint(0, 256, (1, 5), device=DEVICE)
    out_a = eng.generate_tree(ids.clone(), max_new_tokens=4)
    eng._pool.tree_mask_buf.zero_() if eng._pool is not None else None
    out_b = eng.generate_tree(ids.clone(), max_new_tokens=4)
    assert torch.equal(out_a, out_b), "tree-mode output must be deterministic"
    assert eng._pool is not None
    mask = eng._pool.tree_mask_buf[:eng.tree.max_nodes]
    assert mask.numel() > 0
    assert (mask >= 0).all()
    print("[tree-cycle] tree mask written to static pool buffer + deterministic output: OK")


def test_masked_draft_body_direct():
    torch.manual_seed(11)
    mtp = SequentialMTPModule(64, 256, 4).to(DEVICE)
    main = TinyMainModel(64, 256).to(DEVICE)
    eng = SpeculativeDecodingEngine(mtp, main, KVManager(512), temperature=0.0,
                                    depth=4, draft_mode="tree")
    eng._ensure_static_pool(1)
    pool = eng._pool
    cands = eng.tree_builder.top_k_paths(main.proj(main.embed(torch.tensor([[1]], device=DEVICE)))[:, -1, :])
    cand = torch.stack(cands, dim=-1).unsqueeze(1)  # (B, 1, topk)
    active = list(range(min(eng.tree_builder.top_k, 4)))
    m = eng.tree_builder.build_tree_mask(active)
    mf = m[0] if m.dim() == 2 else m
    pool.tree_mask_buf.zero_()
    pool.tree_mask_buf[:mf.numel()].copy_(mf.to(pool.device).float())
    from mtp_engine import draft_tree_masked_body
    # runs the fused masked attention -> norm -> linear path
    draft_tree_masked_body(eng.mtp, pool, eng.tree, cand, pool.tree_mask_buf, 0.0)
    assert pool.draft_tokens[0].numel() > 0
    assert (pool.tree_mask_buf[:eng.tree.max_nodes] >= 0).all()
    print("[tree-cycle] draft_tree_masked_body runs fused attention with mask: OK")


def test_tree_verify_longest_path():
    main_tokens = torch.tensor([[0, 7, 8, 9, 10]], device=DEVICE)
    paths = [[7, 8, 9], [7, 3, 4], [99, 1, 2]]
    mtp = SequentialMTPModule(64, 256, 4).to(DEVICE)
    tb = TreeDraftBuilder(mtp, top_k=2)
    acc, best = verify_tree_branches(None, None, tb.tree, main_tokens, paths)
    assert acc == 3 and best == [7, 8, 9]
    print("[tree-cycle] verify_tree_branches resolves longest accepted path: OK")


def test_tree_mask_passed_to_attention():
    torch.manual_seed(1)
    q = torch.randn(2, 2, 32, device=DEVICE)
    k = torch.randn(2, 2, 8, 32, device=DEVICE)
    v = torch.randn(2, 2, 8, 32, device=DEVICE)
    mask = torch.ones(8, device=DEVICE)
    attn = fused_tree_attention(q, k, v, mask)
    assert attn.shape == (2, 2, 32)
    print("[tree-cycle] fused_tree_attention accepts tree mask: OK")


if __name__ == "__main__":
    test_adaptive_depth_wired()
    test_tree_generation_end_to_end()
    test_tree_mask_coupled_into_attention()
    test_masked_draft_body_direct()
    test_tree_verify_longest_path()
    test_tree_mask_passed_to_attention()
    print("\nALL TREE-CYCLE E2E CHECKS PASSED")
