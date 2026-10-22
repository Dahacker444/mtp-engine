"""Fused speculative-verification kernels.

The verification phase of speculative decoding historically runs as three
separate stages, each with an activation (or control) round-trip:

  1. ``linear(hidden) -> logits``            (a GEMM)
  2. ``argmax(logits) -> tokens``            (a reduction + HBM write)
  3. ``tokens == draft?`` acceptance mask    (a Python loop, ``depth`` times)

``fused_verify`` collapses (1)+(2)+(3) into one kernel for a *single* token.
This module extends that to the **whole draft chain**:

  ``fused_multi_verify`` consumes the batched hidden states of the draft
  sequence, computes logits for every token, argmaxes them, compares the chain
  against the draft tokens, and emits — in a single Triton pass — the per-batch
  accept mask, the number of accepted tokens, and the corrected continuation
  token. No intermediate ``logits``/``tokens`` tensors touch HBM, and there is
  no Python loop over ``depth`` doing per-token comparisons.

Every kernel has a bit-identical eager PyTorch fallback for non-CUDA devices
(e.g. Apple Silicon MPS).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from mtp_kernels import _HAS_TRITON

if _HAS_TRITON:
    import triton
    import triton.language as tl


# --------------------------------------------------------------------------- #
# Single-token fused verify (linear -> argmax -> accept)
# --------------------------------------------------------------------------- #
if _HAS_TRITON:

    @triton.jit
    def fused_verify_kernel(
        Hidden, W, Bias, Draft, Accept, Out, K, V,
        stride_hm, stride_hk,
        stride_wn, stride_wk,
        stride_om, stride_on,
        HAS_BIAS: tl.constexpr,
        BLOCK_V: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        row = tl.program_id(0)

        offs_v = tl.arange(0, BLOCK_V)
        v_mask = offs_v < V

        acc = tl.zeros((BLOCK_V,), dtype=tl.float32)

        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K
            h = tl.load(
                Hidden + row * stride_hm + offs_k * stride_hk,
                mask=k_mask, other=0.0,
            ).to(tl.float32)
            w = tl.load(
                W + offs_v[:, None] * stride_wn + offs_k[None, :] * stride_wk,
                mask=v_mask[:, None] & k_mask[None, :], other=0.0,
            ).to(tl.float32)
            acc += tl.sum(h[None, :] * w, axis=1)

        if HAS_BIAS:
            b = tl.load(Bias + offs_v, mask=v_mask, other=0.0).to(tl.float32)
            acc += b

        top = tl.argmax(acc, axis=0)
        draft = tl.load(Draft + row).to(tl.int32)
        accept = top == draft
        tl.store(Accept + row, accept.to(tl.int8))
        tl.store(Out + row, top.to(tl.int32))


# --------------------------------------------------------------------------- #
# Multi-token chain verification (the fused rollback/commit primitive)
# --------------------------------------------------------------------------- #
if _HAS_TRITON:

    @triton.jit
    def fused_multi_verify_kernel(
        Hidden,        # (B, S, K)  hidden states of the draft sequence
        W,             # (V, K)
        Bias,          # (V,)
        Draft,         # (B, D)     draft token ids for each of D future steps
        Accept,        # (B, D)     per-token acceptance mask (int8)
        AcceptedN,     # (B,)       number of accepted draft tokens
        Corrected,     # (B,)       corrected continuation token (int32)
        B, S, K, V, D,
        stride_hb, stride_hs, stride_hk,
        stride_wn, stride_wk,
        stride_dpwd, stride_dpwd_col,
        stride_acb, stride_acd,
        HAS_BIAS: tl.constexpr,
        BLOCK_V: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        # One program per batch element; walks the S = D+1 sequence tokens and
        # compares each to the corresponding draft token. Loop bound is dynamic
        # (tl.range) so the kernel is captured for any depth.
        b = tl.program_id(0)

        offs_v = tl.arange(0, BLOCK_V)
        v_mask = offs_v < V

        accepted = tl.zeros((), dtype=tl.int32)  # scalar count of accepted drafts

        for s in tl.range(0, S):
            # hidden[s] -> logits[s] (fused GEMM + bias + argmax, online)
            acc = tl.zeros((BLOCK_V,), dtype=tl.float32)
            for k0 in range(0, K, BLOCK_K):
                offs_k = k0 + tl.arange(0, BLOCK_K)
                k_mask = offs_k < K
                h = tl.load(
                    Hidden + b * stride_hb + s * stride_hs + offs_k * stride_hk,
                    mask=k_mask, other=0.0,
                ).to(tl.float32)
                w = tl.load(
                    W + offs_v[:, None] * stride_wn + offs_k[None, :] * stride_wk,
                    mask=v_mask[:, None] & k_mask[None, :], other=0.0,
                ).to(tl.float32)
                acc += tl.sum(h[None, :] * w, axis=1)
            if HAS_BIAS:
                bb = tl.load(Bias + offs_v, mask=v_mask, other=0.0).to(tl.float32)
                acc += bb
            top = tl.argmax(acc, axis=0)

            # Sequence position 0 is the anchor token (verified continuation).
            if s == 0:
                tl.store(Corrected + b, top.to(tl.int32))
            else:
                d_idx = s - 1
                d = tl.load(
                    Draft + b * stride_dpwd + d_idx * stride_dpwd_col,
                    mask=(d_idx < D), other=-1,
                ).to(tl.int32)
                is_accept = (top == d)
                # Only count this draft if the entire prefix was accepted.
                prefix_ok = (accepted == d_idx)
                accept_now = is_accept & prefix_ok
                accept_byte = accept_now.to(tl.int8)
                tl.store(
                    Accept + b * stride_acb + d_idx * stride_acd,
                    accept_byte,
                )
                accepted = tl.where(accept_now, d_idx + 1, accepted)

        tl.store(AcceptedN + b, accepted.to(tl.int32))


def _eager_fused_verify(
    hidden: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor],
    draft: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    logits = torch.nn.functional.linear(hidden.to(torch.float32), weight.to(torch.float32), bias)
    top = logits.argmax(dim=-1).to(torch.int32)
    accept = (top == draft.to(torch.int32))
    return accept.to(torch.int8), top


def _eager_fused_multi_verify(
    hidden: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor],
    draft: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bit-identical eager reference for ``fused_multi_verify``.

    ``hidden`` is ``(B, S, K)``, ``draft`` is ``(B, D)`` with ``S == D + 1``.
    Returns ``(accept, accepted_n, corrected)``.
    """
    B, S, K = hidden.shape
    D = draft.shape[1]
    logits = torch.nn.functional.linear(
        hidden.to(torch.float32), weight.to(torch.float32), bias
    )  # (B, S, V)
    top = logits.argmax(dim=-1).to(torch.int32)  # (B, S)
    accept = torch.zeros((B, D), dtype=torch.int8, device=hidden.device)
    accepted_n = torch.zeros((B,), dtype=torch.int32, device=hidden.device)
    for b in range(B):
        n = 0
        for d in range(D):
            accept[b, d] = (top[b, d + 1] == draft[b, d]).to(torch.int8)
            if accept[b, d]:
                n = d + 1
            else:
                break
        accepted_n[b] = n
    corrected = top[:, 0].to(torch.int32)
    return accept, accepted_n, corrected


def fused_verify(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    draft: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    hidden = hidden.contiguous()
    weight = weight.contiguous()
    draft = draft.contiguous().to(torch.int32)
    M, K = hidden.shape
    V = weight.shape[0]

    if not _HAS_TRITON:
        return _eager_fused_verify(hidden, weight, bias, draft)

    has_bias = bias is not None
    if not has_bias:
        bias = torch.zeros(V, dtype=hidden.dtype, device=hidden.device)
    else:
        bias = bias.contiguous()

    accept = torch.empty(M, dtype=torch.int8, device=hidden.device)
    out = torch.empty(M, dtype=torch.int32, device=hidden.device)

    BLOCK_V = triton.next_power_of_2(V)
    BLOCK_K = 64

    fused_verify_kernel[(M,)](
        hidden, weight, bias, draft, accept, out, K, V,
        hidden.stride(0), hidden.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), 1,  # `out` is 1-D (M,); stride_on is unused
        HAS_BIAS=has_bias,
        BLOCK_V=BLOCK_V, BLOCK_K=BLOCK_K,
    )
    return accept, out


def fused_multi_verify(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    draft: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse linear -> argmax -> chain-acceptance for the whole draft chain.

    Args:
        hidden: ``(B, S, K)`` hidden states of the draft+anchor sequence.
        weight: ``(V, K)`` output projection (the shared lm_head).
        draft:  ``(B, D)`` draft token ids (future steps 1..D).
        bias:   optional ``(V,)``.

    Returns:
        accept    ``(B, D)`` int8 mask (1 = draft token accepted).
        accepted_n ``(B,)`` int32 count of consecutively accepted drafts.
        corrected ``(B,)`` int32 corrected continuation token (argmax at anchor).
    """
    hidden = hidden.contiguous()
    weight = weight.contiguous()
    draft = draft.contiguous().to(torch.int32)
    B, S, K = hidden.shape
    V = weight.shape[0]
    D = draft.shape[1]
    assert S == D + 1, f"expected S == D + 1, got S={S} D={D}"

    if not _HAS_TRITON:
        return _eager_fused_multi_verify(hidden, weight, bias, draft)

    has_bias = bias is not None
    if not has_bias:
        bias = torch.zeros(V, dtype=hidden.dtype, device=hidden.device)
    else:
        bias = bias.contiguous()

    accept = torch.empty((B, D), dtype=torch.int8, device=hidden.device)
    accepted_n = torch.empty((B,), dtype=torch.int32, device=hidden.device)
    corrected = torch.empty((B,), dtype=torch.int32, device=hidden.device)

    BLOCK_V = triton.next_power_of_2(V)
    BLOCK_K = 64

    fused_multi_verify_kernel[(B,)](
        hidden, weight, bias, draft, accept, accepted_n, corrected,
        B, S, K, V, D,
        hidden.stride(0), hidden.stride(1), hidden.stride(2),
        weight.stride(0), weight.stride(1),
        draft.stride(0), draft.stride(1),
        accept.stride(0), accept.stride(1),
        HAS_BIAS=has_bias,
        BLOCK_V=BLOCK_V, BLOCK_K=BLOCK_K,
    )
    return accept, accepted_n, corrected


__all__ = [
    "fused_verify",
    "fused_multi_verify",
    "fused_verify_kernel",
    "fused_multi_verify_kernel",
    "_eager_fused_verify",
    "_eager_fused_multi_verify",
]
