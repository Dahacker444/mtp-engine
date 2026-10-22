from __future__ import annotations

import torch

from mtp_engine import (
    DEVICE,
    SequentialMTPModule,
    SpeculativeDraftTree,
    TreeDraftBuilder,
    ConfidenceScheduler,
    verify_tree_branches,
)
from mtp_kernels import fused_tree_attention


def test_tree_draft_builder():
    torch.manual_seed(0)
    mtp = SequentialMTPModule(64, 256, 4).to(DEVICE)
    tb = TreeDraftBuilder(mtp, top_k=2)
    assert tb.tree.mode == "tree"
    assert tb.tree.max_nodes == 32

    logits = torch.randn(2, 256, device=DEVICE)
    paths = tb.top_k_paths(logits)
    assert len(paths) == 2
    assert all(p.shape == (2,) for p in paths)

    mask = tb.build_tree_mask([0, 1, 2])
    assert mask.shape == (32, 32)
    assert mask.dtype == torch.bool
    assert mask[0, 0].item() is True

    print("[tree-builder] top-k paths + tree mask: OK")


def test_tree_attention_mask():
    torch.manual_seed(1)
    q = torch.randn(2, 2, 32, device=DEVICE)
    k = torch.randn(2, 2, 8, 32, device=DEVICE)
    v = torch.randn(2, 2, 8, 32, device=DEVICE)
    mask = torch.ones(8, device=DEVICE)
    attn = fused_tree_attention(q, k, v, mask)
    assert attn.shape == (2, 2, 32)
    print("[tree-attn] masked attention kernel (eager) runs: OK")


def test_confidence_scheduler():
    cs = ConfidenceScheduler(threshold=0.9, min_depth=1)
    high = torch.tensor([[5.0, 0.0, 0.0]], device=DEVICE)
    low = torch.tensor([[0.0, 0.0, 0.0]], device=DEVICE)
    assert cs.effective_depth(high, 4) == 4
    assert cs.effective_depth(low, 4) == 1
    print("[confidence] high-confidence -> full depth, low -> truncated: OK")


def test_tree_verification():
    main_tokens = torch.tensor([[0, 7, 8, 9, 10]], device=DEVICE)
    paths = [[7, 8, 9], [7, 3, 4], [99, 1, 2]]
    tb = TreeDraftBuilder(SequentialMTPModule(64, 256, 4).to(DEVICE), top_k=2)
    acc, best = verify_tree_branches(None, None, tb.tree, main_tokens, paths)
    assert acc == 3
    assert best == [7, 8, 9]
    print("[verify-tree] longest valid path = [7,8,9] (length 3): OK")


if __name__ == "__main__":
    test_tree_draft_builder()
    test_tree_attention_mask()
    test_confidence_scheduler()
    test_tree_verification()
    print("\nALL TREE / DYNAMIC-LOOKAHEAD CHECKS PASSED")
