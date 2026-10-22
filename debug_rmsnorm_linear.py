"""Minimal standalone repro for the fused_rmsnorm_linear numerical check.

Run directly on the H100 to see the exact values and diff, isolating whether
the Triton kernel matches its eager reference:

    CUDA_VISIBLE_DEVICES=0 python debug_rmsnorm_linear.py
"""

import math

import torch

from mtp_kernels import (
    fused_rmsnorm_linear,
    _eager_rmsnorm_linear,
    _HAS_TRITON,
    TUNING,
)

torch.manual_seed(0)
x = torch.randn(5, 64, device="cuda")
W = torch.randn(128, 64, device="cuda")
b = torch.randn(128, device="cuda")

print(f"HAS_TRITON={_HAS_TRITON}")
print(f"TUNING.fp8_scale={TUNING.fp8_scale}")
print(f"TUNING.BLOCK_M={TUNING.BLOCK_M} BLOCK_N={TUNING.BLOCK_N} BLOCK_K={TUNING.BLOCK_K}")

# The public fused path (Triton kernel on CUDA).
r_lin = fused_rmsnorm_linear(x, W, b, 1e-6)

# The eager reference (bit-identical fallback).
r_eager = _eager_rmsnorm_linear(x, W, b, 1e-6, TUNING.fp8_scale)

# The torch reference used in mtp_engine.py's assert.
xn = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-6)
r_ref = torch.nn.functional.linear(xn, W, b)

print(f"fused   shape={tuple(r_lin.shape)}  [0,0]={r_lin[0,0].item():.6f}")
print(f"eager   shape={tuple(r_eager.shape)}  [0,0]={r_eager[0,0].item():.6f}")
print(f"torch   shape={tuple(r_ref.shape)}   [0,0]={r_ref[0,0].item():.6f}")

d1 = (r_lin - r_eager).abs().max().item()
d2 = (r_lin - r_ref).abs().max().item()
d3 = (r_eager - r_ref).abs().max().item()
print(f"max |fused - eager| = {d1:.3e}")
print(f"max |fused - torch| = {d2:.3e}")
print(f"max |eager - torch| = {d3:.3e}")
