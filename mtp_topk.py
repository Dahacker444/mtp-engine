from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from mtp_engine import SpeculativeDraftTree


def top_k_probs(logits: torch.Tensor, k: int, temperature: float) -> Tuple[torch.Tensor, torch.Tensor]:
    l = logits / temperature if temperature > 0 else logits
    probs = torch.softmax(l, dim=-1)
    topk_vals, topk_idx = torch.topk(probs, k=k, dim=-1)
    return topk_vals, topk_idx


def sample_top_k(logits: torch.Tensor, k: int, temperature: float) -> torch.Tensor:
    if temperature <= 0.0:
        return logits.argmax(dim=-1)
    k = min(k, logits.shape[-1])
    topk_vals, topk_idx = torch.topk(logits / temperature, k=k, dim=-1)
    topk_probs = torch.softmax(topk_vals, dim=-1)
    sampled = torch.multinomial(topk_probs, num_samples=1).squeeze(-1)
    return torch.gather(topk_idx, -1, sampled.unsqueeze(-1)).squeeze(-1)


def build_tree_mask(
    tree: SpeculativeDraftTree, node_idx: torch.Tensor
) -> torch.Tensor:
    B = node_idx.shape[0]
    L = tree.max_nodes
    mask = torch.zeros((B, L), dtype=torch.float32, device=tree.device)
    for b in range(B):
        n = int(node_idx[b].item())
        if n < 0 or n >= L:
            continue
        mask[b] = tree.tree_mask[n]
    return mask


def expand_branches(
    tree: SpeculativeDraftTree, tokens: torch.Tensor, k: int
) -> List[List[int]]:
    paths: List[List[int]] = []
    for b in range(tokens.shape[0]):
        row = tokens[b].tolist()
        if not row:
            paths.append([])
            continue
        parent = 0
        branch_path: List[int] = [row[0]]
        for t in row[1:]:
            branch_path.append(t)
            if len(branch_path) > tree.max_branches:
                break
        paths.append(branch_path[: min(k, len(branch_path))])
    return paths


__all__ = ["top_k_probs", "sample_top_k", "build_tree_mask", "expand_branches"]
