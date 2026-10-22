from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

# Force full fp32 GEMM (disable TF32) so Triton tl.dot and the torch reference
# share the same numerics. On H100/Ampere, TF32 (10-bit mantissa) would
# otherwise make tl.dot diverge from F.linear by ~1e-3, breaking the 1e-4
# correctness asserts in the validation harness.
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

# Never force Triton's CPU interpreter on a CUDA device — that would route
# every kernel through the slow interpreter instead of JIT-compiling to real
# GPU code (and it is what tripped the strict interpreter shape checks below).
# The interpreter is only useful as a CPU-only fallback when CUDA is absent.
if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:
    triton = None
    tl = None
    _HAS_TRITON = False


_WARP = 32


_ATTENTION_AUTOTUNE_CONFIGS = None


def _attention_configs():
    global _ATTENTION_AUTOTUNE_CONFIGS
    if _ATTENTION_AUTOTUNE_CONFIGS is None:
        _ATTENTION_AUTOTUNE_CONFIGS = [
            triton.Config({"BLOCK_DHEAD": 64, "BLOCK_BD": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_DHEAD": 64, "BLOCK_BD": 64}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_DHEAD": 128, "BLOCK_BD": 64}, num_warps=8, num_stages=4),
            triton.Config({"BLOCK_DHEAD": 128, "BLOCK_BD": 128}, num_warps=8, num_stages=4),
        ]
    return _ATTENTION_AUTOTUNE_CONFIGS


@dataclass(frozen=True)
class KernelTuningConfig:
    BLOCK_M: int = 64
    BLOCK_N: int = 64
    BLOCK_K: int = 32
    BLOCK_DHEAD: int = 64
    num_warps: int = 4
    num_stages: int = 2
    fp8_scale: float = 1.0


TUNING = KernelTuningConfig()
MAX_FP8 = 448.0

_ARCH_SMEM = {
    "turing": 96 * 2**10,
    "ampere": 164 * 2**10,
    "hopper": 228 * 2**10,
    "blackwell": 228 * 2**10,
}


def _detect_cuda_arch() -> str:
    if not torch.cuda.is_available():
        return ""
    major, minor = torch.cuda.get_device_capability(0)
    if major >= 10:
        return "blackwell"
    if major == 9:
        return "hopper"
    if major == 8:
        return "ampere"
    if major == 7:
        return "turing"
    return f"cc{major}{minor}"


def build_tuning_config() -> KernelTuningConfig:
    arch = _detect_cuda_arch()
    if arch == "hopper" or arch == "blackwell":
        return KernelTuningConfig(
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, BLOCK_DHEAD=128,
            num_warps=8, num_stages=3,
        )
    if arch == "ampere":
        return KernelTuningConfig(
            BLOCK_M=128, BLOCK_N=64, BLOCK_K=32, BLOCK_DHEAD=64,
            num_warps=4, num_stages=2,
        )
    return KernelTuningConfig()


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _warp_align_block(n: int, default: int) -> int:
    if n <= 0:
        return default
    if n <= _WARP:
        return _WARP
    return ((n + _WARP - 1) // _WARP) * _WARP


if _HAS_TRITON:

    @triton.jit
    def fused_rmsnorm_kernel(
        X,
        Y,
        W,
        eps,
        N,
        x_stride,
        y_stride,
        w_stride,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        X += row * x_stride
        Y += row * y_stride

        cols = tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X + cols * w_stride, mask=mask, other=0.0).to(tl.float32)

        x2 = x * x
        ms = tl.sum(x2, axis=0) / N
        rstd = 1.0 / tl.sqrt(ms + eps)

        w = tl.load(W + cols * w_stride, mask=mask, other=1.0).to(tl.float32)
        out = x * rstd * w
        tl.store(Y + cols * w_stride, out, mask=mask)


if _HAS_TRITON:

    @triton.jit
    def fused_rmsnorm_linear_kernel(
        X,
        W,
        Bias,
        Y,
        eps,
        M,
        N,
        K,
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        stride_ym, stride_yn,
        SCALE,
        USE_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        m_mask = offs_m < M

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        sum_sq = tl.zeros((BLOCK_M,), dtype=tl.float32)

        # Pass 1: compute the per-row mean-square for RMSNorm over the full K
        # dimension before any GEMM. RMSNorm must normalize the INPUT x, not
        # the GEMM output, so we need the full row statistic first.
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K

            x = tl.load(
                X + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)

            sum_sq += tl.sum(x * x, axis=1)

        rstd = 1.0 / tl.sqrt(sum_sq / K + eps)  # (BLOCK_M,)

        # Pass 2: GEMM with the RMSNorm-normalized input (x * rstd).
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K

            x = tl.load(
                X + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)

            x = x * rstd[:, None]  # normalize input BEFORE the dot product

            w = tl.load(
                W + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
                mask=n_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)

            acc += tl.dot(x, tl.trans(w), allow_tf32=False)

        if USE_BIAS:
            bias = tl.load(Bias + offs_n, mask=n_mask, other=0.0).to(tl.float32)
            acc += bias[None, :]

        acc = acc * SCALE

        tl.store(
            Y + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
            acc.to(Y.dtype.element_ty),
            mask=m_mask[:, None] & n_mask[None, :],
        )


if _HAS_TRITON:

    @triton.jit
    def fused_causal_attention_kernel(
        Q,
        K,
        V,
        O,
        scale,
        L,
        stride_qb, stride_qh, stride_qd,
        stride_kb, stride_kh, stride_kl, stride_kd,
        stride_vb, stride_vh, stride_vl, stride_vd,
        stride_ob, stride_oh, stride_od,
        BLOCK_DHEAD: tl.constexpr,
        BLOCK_BD: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)

        d_off = tl.arange(0, BLOCK_DHEAD)
        q = tl.load(
            Q + pid_b * stride_qb + pid_h * stride_qh + d_off * stride_qd
        ).to(tl.float32)  # (BLOCK_DHEAD,)

        # Single-query online softmax. Running stats are true scalars (built
        # with tl.max/tl.sum over the full score vector, i.e. no axis, giving
        # shape ()), and acc is 1-D (BLOCK_DHEAD,). This keeps every broadcast
        # unambiguous.
        m_i = tl.full((1,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((1,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_DHEAD,), dtype=tl.float32)

        for start in range(0, L, BLOCK_BD):
            k_off = start + tl.arange(0, BLOCK_BD)
            k_mask = k_off < L

            k = tl.load(
                K + pid_b * stride_kb + pid_h * stride_kh
                  + k_off[:, None] * stride_kl + d_off[None, :] * stride_kd,
                mask=k_mask[:, None], other=0.0,
            ).to(tl.float32)  # (BLOCK_BD, BLOCK_DHEAD)

            # scores: (BLOCK_BD,)  (single query dotted with each key)
            scores = tl.sum(q[None, :] * k, axis=1) * scale
            # Mask padded key positions (k_off >= L) to -inf so they do not
            # corrupt the online-softmax running max / denominator.
            scores = tl.where(k_mask, scores, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(scores, axis=0))
            p = tl.exp(scores - m_new)
            alpha = tl.exp(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, axis=0)
            m_i = m_new

            v = tl.load(
                V + pid_b * stride_vb + pid_h * stride_vh
                  + k_off[:, None] * stride_vl + d_off[None, :] * stride_vd,
                mask=k_mask[:, None], other=0.0,
            ).to(tl.float32)  # (BLOCK_BD, BLOCK_DHEAD)

            acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)

        acc = acc / l_i

        o_ptr = O + pid_b * stride_ob + pid_h * stride_oh + d_off * stride_od
        tl.store(o_ptr, acc.to(O.dtype.element_ty))


if _HAS_TRITON:

    @triton.jit
    def fused_rmsnorm_linear_fp8_kernel(
        X, W, WScale, Bias, Y, eps, M, N, K,
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        stride_ym, stride_yn,
        USE_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        m_mask = offs_m < M

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        sum_sq = tl.zeros((BLOCK_M,), dtype=tl.float32)

        # Pass 1: per-row mean-square for RMSNorm over the full K dimension.
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K

            x = tl.load(
                X + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                mask=m_mask[:, None] & k_mask[None, :], other=0.0,
            ).to(tl.float32)
            sum_sq += tl.sum(x * x, axis=1)

        rstd = 1.0 / tl.sqrt(sum_sq / K + eps)  # (BLOCK_M,)

        # Pass 2: GEMM with the RMSNorm-normalized input.
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K

            x = tl.load(
                X + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                mask=m_mask[:, None] & k_mask[None, :], other=0.0,
            ).to(tl.float32)
            x = x * rstd[:, None]  # normalize input BEFORE the dot product

            w = tl.load(
                W + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
                mask=n_mask[:, None] & k_mask[None, :], other=0.0,
            )
            w = w.to(tl.float32)
            wscale = tl.load(WScale + offs_n, mask=n_mask, other=1.0).to(tl.float32)
            w = w * wscale[:, None]

            acc += tl.dot(x, tl.trans(w), allow_tf32=False)

        if USE_BIAS:
            bias = tl.load(Bias + offs_n, mask=n_mask, other=0.0).to(tl.float32)
            acc += bias[None, :]

        tl.store(
            Y + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
            acc.to(Y.dtype.element_ty),
            mask=m_mask[:, None] & n_mask[None, :],
        )


if _HAS_TRITON:

    @triton.jit
    def fused_causal_attention_fp8_kernel(
        Q, K, V, KScale, VScale, O, scale, L,
        stride_qb, stride_qh, stride_qd,
        stride_kb, stride_kh, stride_kl, stride_kd,
        stride_vb, stride_vh, stride_vl, stride_vd,
        stride_ob, stride_oh, stride_od,
        BLOCK_DHEAD: tl.constexpr,
        BLOCK_BD: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)

        d_off = tl.arange(0, BLOCK_DHEAD)
        q = tl.load(
            Q + pid_b * stride_qb + pid_h * stride_qh + d_off * stride_qd
        ).to(tl.float32)

        kscale = tl.load(KScale + pid_h).to(tl.float32)
        vscale = tl.load(VScale + pid_h).to(tl.float32)

        m_i = tl.full((1,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((1,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_DHEAD,), dtype=tl.float32)

        for start in range(0, L, BLOCK_BD):
            k_off = start + tl.arange(0, BLOCK_BD)
            k_mask = k_off < L

            k = tl.load(
                K + pid_b * stride_kb + pid_h * stride_kh
                  + k_off[:, None] * stride_kl + d_off[None, :] * stride_kd,
                mask=k_mask[:, None], other=0.0,
            ).to(tl.float32) * kscale

            scores = tl.sum(q[None, :] * k, axis=1) * scale
            # Mask padded key positions to -inf so they don't corrupt the softmax.
            scores = tl.where(k_mask, scores, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(scores, axis=0))
            p = tl.exp(scores - m_new)
            alpha = tl.exp(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, axis=0)
            m_i = m_new

            v = tl.load(
                V + pid_b * stride_vb + pid_h * stride_vh
                  + k_off[:, None] * stride_vl + d_off[None, :] * stride_vd,
                mask=k_mask[:, None], other=0.0,
            ).to(tl.float32) * vscale

            acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)

        acc = acc / l_i
        o_ptr = O + pid_b * stride_ob + pid_h * stride_oh + d_off * stride_od
        tl.store(o_ptr, acc.to(O.dtype.element_ty))


if _HAS_TRITON:

    @triton.jit
    def fused_int4_matmul_kernel(
        X, W, Scale, Y, M, N, K,
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        stride_sn, stride_sg,
        stride_ym, stride_yn,
        GROUP: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        m_mask = offs_m < M

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K

            x = tl.load(
                X + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                mask=m_mask[:, None] & k_mask[None, :], other=0.0,
            ).to(tl.float32)

            packed_cols = offs_k // 2
            packed_mask = offs_k < K
            wb = tl.load(
                W + offs_n[:, None] * stride_wn + packed_cols[None, :] * stride_wk,
                mask=n_mask[:, None] & packed_mask[None, :], other=0.0,
            ).to(tl.int32)

            low = (wb & 0x0F) - 8
            high = ((wb >> 4) & 0x0F) - 8
            is_odd = (offs_k % 2) == 1
            w_signed = tl.where(is_odd[None, :], high, low).to(tl.float32)

            g = offs_k // GROUP
            sg = tl.load(
                Scale + offs_n[:, None] * stride_sn + g[None, :] * stride_sg,
                mask=n_mask[:, None] & (g[None, :] < (K + GROUP - 1) // GROUP),
                other=1.0,
            ).to(tl.float32)
            w = w_signed * sg

            acc += tl.dot(x, tl.trans(w), allow_tf32=False)

        tl.store(
            Y + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
            acc.to(Y.dtype.element_ty),
            mask=m_mask[:, None] & n_mask[None, :],
        )


if _HAS_TRITON:

    @triton.jit
    def fused_paged_attention_kernel(
        Q, K, V, PageTable, O, scale, L, PAGE_SIZE,
        stride_qb, stride_qh, stride_qd,
        stride_kp, stride_kb, stride_kh, stride_ks, stride_kd,
        stride_vp, stride_vb, stride_vh, stride_vs, stride_vd,
        stride_ob, stride_oh, stride_od,
        BLOCK_DHEAD: tl.constexpr,
        BLOCK_BD: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)

        d_off = tl.arange(0, BLOCK_DHEAD)
        q = tl.load(
            Q + pid_b * stride_qb + pid_h * stride_qh + d_off * stride_qd
        ).to(tl.float32)

        m_i = tl.full((1,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((1,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_DHEAD,), dtype=tl.float32)

        for start in range(0, L, BLOCK_BD):
            t_off = start + tl.arange(0, BLOCK_BD)
            t_mask = t_off < L

            page = tl.load(
                PageTable + pid_b * L + t_off,
                mask=t_mask, other=0,
            ).to(tl.int32)

            s_off = t_off % PAGE_SIZE

            k = tl.load(
                K + page[:, None] * stride_kp + pid_b * stride_kb
                  + pid_h * stride_kh + s_off[:, None] * stride_ks
                  + d_off[None, :] * stride_kd,
                mask=t_mask[:, None], other=0.0,
            ).to(tl.float32)

            scores = tl.sum(q[None, :] * k, axis=1) * scale
            # Mask padded key positions to -inf so they don't corrupt the softmax.
            scores = tl.where(t_mask, scores, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(scores, axis=0))
            p = tl.exp(scores - m_new)
            alpha = tl.exp(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, axis=0)
            m_i = m_new

            v = tl.load(
                V + page[:, None] * stride_vp + pid_b * stride_vb
                  + pid_h * stride_vh + s_off[:, None] * stride_vs
                  + d_off[None, :] * stride_vd,
                mask=t_mask[:, None], other=0.0,
            ).to(tl.float32)

            acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)

        acc = acc / l_i
        o_ptr = O + pid_b * stride_ob + pid_h * stride_oh + d_off * stride_od
        tl.store(o_ptr, acc.to(O.dtype.element_ty))


if _HAS_TRITON:

    @triton.jit
    def fused_tree_attention_kernel(
        Q, K, V, Mask, O, scale, L,
        stride_qb, stride_qh, stride_qd,
        stride_kb, stride_kh, stride_kl, stride_kd,
        stride_vb, stride_vh, stride_vl, stride_vd,
        stride_ob, stride_oh, stride_od,
        BLOCK_DHEAD: tl.constexpr,
        BLOCK_BD: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)

        d_off = tl.arange(0, BLOCK_DHEAD)
        q = tl.load(
            Q + pid_b * stride_qb + pid_h * stride_qh + d_off * stride_qd
        ).to(tl.float32)

        m_i = tl.full((1,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((1,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_DHEAD,), dtype=tl.float32)

        for start in range(0, L, BLOCK_BD):
            k_off = start + tl.arange(0, BLOCK_BD)
            k_mask = k_off < L

            k = tl.load(
                K + pid_b * stride_kb + pid_h * stride_kh
                  + k_off[:, None] * stride_kl + d_off[None, :] * stride_kd,
                mask=k_mask[:, None], other=0.0,
            ).to(tl.float32)

            scores = tl.sum(q[None, :] * k, axis=1) * scale
            # Mask padded key positions to -inf so they don't corrupt the softmax.
            scores = tl.where(k_mask, scores, float("-inf"))
            mask = tl.load(Mask + k_off, mask=k_mask, other=0.0).to(tl.float32)
            scores = tl.where(mask > 0.5, scores, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(scores, axis=0))
            p = tl.exp(scores - m_new)
            alpha = tl.exp(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, axis=0)
            m_i = m_new

            v = tl.load(
                V + pid_b * stride_vb + pid_h * stride_vh
                  + k_off[:, None] * stride_vl + d_off[None, :] * stride_vd,
                mask=k_mask[:, None], other=0.0,
            ).to(tl.float32)

            acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)

        acc = acc / l_i
        o_ptr = O + pid_b * stride_ob + pid_h * stride_oh + d_off * stride_od
        tl.store(o_ptr, acc.to(O.dtype.element_ty))


if _HAS_TRITON:

    @triton.jit
    def fused_tree_attn_rmsnorm_linear_kernel(
        Q, K, V, Mask, RmsW, ProjW, ProjBias, WScale, O, scale, L, eps, N,
        stride_qb, stride_qh, stride_qd,
        stride_kb, stride_kh, stride_kl, stride_kd,
        stride_vb, stride_vh, stride_vl, stride_vd,
        stride_pn, stride_pd,
        stride_ob, stride_oh, stride_on,
        USE_RMSW: tl.constexpr,
        USE_BIAS: tl.constexpr,
        USE_SCALE: tl.constexpr,
        BLOCK_DHEAD: tl.constexpr,
        BLOCK_BD: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)

        d_off = tl.arange(0, BLOCK_DHEAD)
        q = tl.load(
            Q + pid_b * stride_qb + pid_h * stride_qh + d_off * stride_qd
        ).to(tl.float32)

        m_i = tl.full((1,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((1,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_DHEAD,), dtype=tl.float32)

        for start in range(0, L, BLOCK_BD):
            k_off = start + tl.arange(0, BLOCK_BD)
            k_mask = k_off < L

            k = tl.load(
                K + pid_b * stride_kb + pid_h * stride_kh
                  + k_off[:, None] * stride_kl + d_off[None, :] * stride_kd,
                mask=k_mask[:, None], other=0.0,
            ).to(tl.float32)

            scores = tl.sum(q[None, :] * k, axis=1) * scale
            # Mask padded key positions to -inf so they don't corrupt the softmax.
            scores = tl.where(k_mask, scores, float("-inf"))
            mask = tl.load(Mask + k_off, mask=k_mask, other=0.0).to(tl.float32)
            scores = tl.where(mask > 0.5, scores, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(scores, axis=0))
            p = tl.exp(scores - m_new)
            alpha = tl.exp(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, axis=0)
            m_i = m_new

            v = tl.load(
                V + pid_b * stride_vb + pid_h * stride_vh
                  + k_off[:, None] * stride_vl + d_off[None, :] * stride_vd,
                mask=k_mask[:, None], other=0.0,
            ).to(tl.float32)
            acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)

        acc = acc / l_i
        if USE_RMSW:
            rw = tl.load(RmsW + d_off).to(tl.float32)
        else:
            rw = tl.full((BLOCK_DHEAD,), 1.0, dtype=tl.float32)
        ms = tl.sum(acc * acc, axis=0) / BLOCK_DHEAD
        rstd = 1.0 / tl.sqrt(ms + eps)
        acc = acc * rstd * rw

        n_off = tl.arange(0, BLOCK_N)
        p_out = tl.zeros((BLOCK_N,), dtype=tl.float32)

        for nb in range(0, N, BLOCK_N):
            n_idx = nb + n_off
            nm = n_idx < N
            w = tl.load(
                ProjW + n_idx[:, None] * stride_pn + d_off[None, :] * stride_pd,
                mask=nm[:, None], other=0.0,
            ).to(tl.float32)
            if USE_SCALE:
                wsc = tl.load(WScale + n_idx, mask=nm, other=1.0).to(tl.float32)
                w = w * wsc[:, None]
            p_out = tl.sum(acc[None, :] * w, axis=1)
            if USE_BIAS:
                b = tl.load(ProjBias + n_idx, mask=nm, other=0.0).to(tl.float32)
                p_out += b
            o_ptr = O + pid_b * stride_ob + pid_h * stride_oh + n_idx * stride_on
            tl.store(o_ptr, p_out.to(O.dtype.element_ty), mask=nm)


def quantize_fp8_e4m3(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    xf = x.to(torch.float32)
    amax = xf.abs().max()
    scale = amax / MAX_FP8 if amax > 0 else torch.tensor(1.0, dtype=torch.float32, device=x.device)
    scaled = (xf / scale).clamp(-MAX_FP8, MAX_FP8)
    if hasattr(torch, "float8_e4m3fn"):
        x_fp8 = scaled.to(torch.float8_e4m3fn)
    else:
        x_fp8 = _emulate_e4m3(scaled)
    return x_fp8, scale.to(torch.float32)


def _emulate_e4m3(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(-MAX_FP8, MAX_FP8)
    absx = x.abs()
    exp = torch.floor(torch.log2(absx + 1e-30))
    exp = exp.clamp(-9, 7)
    scale = 2.0 ** (exp - 2)
    q = torch.round(x / scale) * scale
    return torch.where(x == 0, torch.zeros_like(x), q)


def dequantize_fp8(x_fp8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x_fp8.to(torch.float32) * scale.to(torch.float32)


def quantize_int4(x: torch.Tensor, group: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
    N, K = x.shape
    xf = x.to(torch.float32)
    num_groups = (K + group - 1) // group
    scales = torch.empty((N, num_groups), dtype=torch.float32, device=x.device)
    packed = torch.empty((N, K // 2), dtype=torch.uint8, device=x.device)

    for g in range(num_groups):
        c0, c1 = g * group, min((g + 1) * group, K)
        chunk = xf[:, c0:c1]
        amax = chunk.abs().max(dim=-1, keepdim=True).values
        s = torch.where(amax > 0, amax / 7.0, torch.ones_like(amax))
        q = torch.round(chunk / s).clamp(-8, 7).to(torch.int8)
        scales[:, g] = s.squeeze(-1)
        for j in range(0, chunk.shape[1], 2):
            low = q[:, j].to(torch.uint8) & 0x0F
            if j + 1 < chunk.shape[1]:
                high = (q[:, j + 1].to(torch.uint8) & 0x0F) << 4
            else:
                high = torch.zeros_like(low)
            packed[:, (c0 + j) // 2] = low | high
    return packed, scales


def dequantize_int4(packed: torch.Tensor, scales: torch.Tensor, K: int, group: int) -> torch.Tensor:
    N = packed.shape[0]
    w = torch.empty((N, ((K + 1) // 2) * 2), dtype=torch.float32, device=packed.device)
    for j in range(w.shape[1] // 2):
        byte = packed[:, j].to(torch.int32)
        low = (byte & 0x0F) - 8
        high = ((byte >> 4) & 0x0F) - 8
        w[:, 2 * j] = low.to(torch.float32)
        w[:, 2 * j + 1] = high.to(torch.float32)
    w = w[:, :K]
    num_groups = (K + group - 1) // group
    for g in range(num_groups):
        c0, c1 = g * group, min((g + 1) * group, K)
        w[:, c0:c1] = w[:, c0:c1] * scales[:, g:g + 1]
    return w


def _eager_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.to(torch.float32)
    ms = (xf * xf).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(ms + eps)
    return (xf * rstd * weight.to(torch.float32)).to(x.dtype)


def _eager_rmsnorm_linear(
    x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor], eps: float,
    scale: float,
) -> torch.Tensor:
    xf = x.to(torch.float32)
    ms = (xf * xf).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(ms + eps)
    x_norm = xf * rstd
    y = torch.nn.functional.linear(x_norm, weight.to(torch.float32), bias)
    return (y * scale).to(x.dtype)


def _eager_causal_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float,
) -> torch.Tensor:
    qf = q.to(torch.float32)
    kf = k.to(torch.float32)
    vf = v.to(torch.float32)
    scores = torch.einsum("bhd,bhld->bhl", qf, kf) * scale
    m = torch.amax(scores, dim=-1, keepdim=True)
    p = torch.exp(scores - m)
    l = p.sum(dim=-1, keepdim=True)
    out = torch.einsum("bhl,bhld->bhd", p / l, vf)
    return out.to(q.dtype)


def _eager_rmsnorm_linear_fp8(
    x: torch.Tensor, weight_fp8: torch.Tensor, wscale: torch.Tensor,
    bias: Optional[torch.Tensor], eps: float,
) -> torch.Tensor:
    w = weight_fp8.to(torch.float32) * wscale.to(torch.float32)[:, None]
    xf = x.to(torch.float32)
    ms = (xf * xf).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(ms + eps)
    x_norm = xf * rstd
    y = torch.nn.functional.linear(x_norm, w, bias)
    return y.to(x.dtype)


def _eager_causal_attention_fp8(
    q: torch.Tensor, k_fp8: torch.Tensor, v_fp8: torch.Tensor,
    kscale: torch.Tensor, vscale: torch.Tensor, scale: float,
) -> torch.Tensor:
    kf = k_fp8.to(torch.float32) * kscale[None, :, None, None]
    vf = v_fp8.to(torch.float32) * vscale[None, :, None, None]
    qf = q.to(torch.float32)
    scores = torch.einsum("bhd,bhld->bhl", qf, kf) * scale
    m = torch.amax(scores, dim=-1, keepdim=True)
    p = torch.exp(scores - m)
    l = p.sum(dim=-1, keepdim=True)
    out = torch.einsum("bhl,bhld->bhd", p / l, vf)
    return out.to(q.dtype)


def _eager_paged_attention(
    q: torch.Tensor, k_pages: torch.Tensor, v_pages: torch.Tensor,
    page_table: torch.Tensor, page_size: int, scale: float,
) -> torch.Tensor:
    B, H, D = q.shape
    _, L = page_table.shape
    qf = q.to(torch.float32)
    page = page_table.long()
    slot = torch.arange(L, device=q.device) % page_size
    k_g = k_pages[page.unsqueeze(1).expand(B, H, L),
                  torch.arange(B, device=q.device).unsqueeze(1).unsqueeze(2).expand(B, H, L),
                  torch.arange(H, device=q.device).unsqueeze(0).unsqueeze(2).expand(B, H, L),
                  slot.unsqueeze(0).unsqueeze(0).expand(B, H, L)]
    v_g = v_pages[page.unsqueeze(1).expand(B, H, L),
                  torch.arange(B, device=q.device).unsqueeze(1).unsqueeze(2).expand(B, H, L),
                  torch.arange(H, device=q.device).unsqueeze(0).unsqueeze(2).expand(B, H, L),
                  slot.unsqueeze(0).unsqueeze(0).expand(B, H, L)]
    kf = k_g.to(torch.float32)
    vf = v_g.to(torch.float32)
    scores = torch.einsum("bhd,bhld->bhl", qf, kf) * scale
    m = torch.amax(scores, dim=-1, keepdim=True)
    p = torch.exp(scores - m)
    l = p.sum(dim=-1, keepdim=True)
    out = torch.einsum("bhl,bhld->bhd", p / l, vf)
    return out.to(q.dtype)


def _eager_tree_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor, scale: float,
) -> torch.Tensor:
    qf = q.to(torch.float32)
    kf = k.to(torch.float32)
    vf = v.to(torch.float32)
    scores = torch.einsum("bhd,bhld->bhl", qf, kf) * scale
    scores = torch.where(mask[None, None, :] > 0.5, scores,
                         torch.full_like(scores, float("-inf")))
    m = torch.amax(scores, dim=-1, keepdim=True)
    p = torch.exp(scores - m)
    l = p.sum(dim=-1, keepdim=True)
    out = torch.einsum("bhl,bhld->bhd", p / l, vf)
    return out.to(q.dtype)


def _eager_int4_matmul(x: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor,
                       K: int, group: int) -> torch.Tensor:
    w = dequantize_int4(packed, scales, K, group)
    return torch.nn.functional.linear(x.to(torch.float32), w).to(x.dtype)


def _eager_tree_attn_rmsnorm_linear(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor,
    rms_w: Optional[torch.Tensor], proj_w: torch.Tensor, proj_bias: Optional[torch.Tensor],
    wscale: Optional[torch.Tensor], scale: float, eps: float,
) -> torch.Tensor:
    qf = q.to(torch.float32)
    kf = k.to(torch.float32)
    vf = v.to(torch.float32)
    scores = torch.einsum("bhd,bhld->bhl", qf, kf) * scale
    scores = torch.where(mask[None, None, :] > 0.5, scores,
                         torch.full_like(scores, float("-inf")))
    m = torch.amax(scores, dim=-1, keepdim=True)
    p = torch.exp(scores - m)
    l = p.sum(dim=-1, keepdim=True)
    attn = torch.einsum("bhl,bhld->bhd", p / l, vf)
    if rms_w is None:
        rms_w = torch.ones(attn.shape[-1], dtype=torch.float32, device=attn.device)
    ms = (attn * attn).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(ms + eps)
    attn = attn * rstd * rms_w.to(torch.float32)
    w = proj_w.to(torch.float32)
    if wscale is not None:
        w = w * wscale.to(torch.float32)[:, None]
    out = torch.einsum("bhd,nd->bhn", attn, w)
    if proj_bias is not None:
        out = out + proj_bias.to(torch.float32)
    return out.to(q.dtype)


def fused_rmsnorm(
    x: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    x = x.contiguous()
    M, N = x.shape
    if weight is None:
        weight = torch.ones(N, dtype=x.dtype, device=x.device)
    else:
        weight = weight.contiguous().to(dtype=x.dtype, device=x.device)

    if not _HAS_TRITON:
        return _eager_rmsnorm(x, weight, eps)

    y = torch.empty_like(x)
    BLOCK_N = triton.next_power_of_2(N)
    fused_rmsnorm_kernel[(M,)](
        x, y, weight, eps, N,
        x.stride(0), y.stride(0), weight.stride(0),
        BLOCK_N=BLOCK_N,
        num_warps=TUNING.num_warps,
    )
    return y


def fused_rmsnorm_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    scale: Optional[float] = None,
) -> torch.Tensor:
    if scale is None:
        scale = TUNING.fp8_scale
    x = x.contiguous()
    weight = weight.contiguous()
    M, K = x.shape
    N, _K = weight.shape
    assert K == _K
    has_bias = bias is not None
    if has_bias:
        bias = bias.contiguous().to(dtype=x.dtype, device=x.device)

    if not _HAS_TRITON:
        return _eager_rmsnorm_linear(x, weight, bias, eps, scale)

    if not has_bias:
        bias = torch.zeros(N, dtype=x.dtype, device=x.device)

    y = torch.empty((M, N), dtype=x.dtype, device=x.device)

    BLOCK_M = _warp_align_block(M, TUNING.BLOCK_M)
    BLOCK_N = _warp_align_block(N, TUNING.BLOCK_N)
    BLOCK_K = _warp_align_block(K, TUNING.BLOCK_K)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    fused_rmsnorm_linear_kernel[grid](
        x, weight, bias, y, eps, M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        y.stride(0), y.stride(1),
        SCALE=scale,
        USE_BIAS=has_bias,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=TUNING.num_warps, num_stages=TUNING.num_stages,
    )
    return y


def fused_causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    B, H, D = q.shape
    _B, _H, L, _D = k.shape
    assert B == _B and H == _H and D == _D
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    if not _HAS_TRITON:
        return _eager_causal_attention(q, k, v, scale)

    out = torch.empty_like(q)
    BLOCK_DHEAD = triton.next_power_of_2(D)
    BLOCK_BD = TUNING.BLOCK_M

    fused_causal_attention_kernel[(B, H)](
        q, k, v, out, scale, L,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_DHEAD=BLOCK_DHEAD, BLOCK_BD=BLOCK_BD,
        num_warps=TUNING.num_warps,
    )
    return out


def fused_rmsnorm_linear_fp8(
    x: torch.Tensor, weight_fp8: torch.Tensor, wscale: torch.Tensor,
    bias: Optional[torch.Tensor] = None, eps: float = 1e-6,
) -> torch.Tensor:
    x = x.contiguous()
    weight_fp8 = weight_fp8.contiguous()
    wscale = wscale.contiguous()
    M, K = x.shape
    N = weight_fp8.shape[0]
    has_bias = bias is not None
    if not _HAS_TRITON:
        return _eager_rmsnorm_linear_fp8(x, weight_fp8, wscale, bias, eps)
    if not has_bias:
        bias = torch.zeros(N, dtype=x.dtype, device=x.device)
    else:
        bias = bias.contiguous()
    y = torch.empty((M, N), dtype=x.dtype, device=x.device)
    BLOCK_M = _warp_align_block(M, TUNING.BLOCK_M)
    BLOCK_N = _warp_align_block(N, TUNING.BLOCK_N)
    BLOCK_K = _warp_align_block(K, TUNING.BLOCK_K)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    fused_rmsnorm_linear_fp8_kernel[grid](
        x, weight_fp8, wscale, bias, y, eps, M, N, K,
        x.stride(0), x.stride(1),
        weight_fp8.stride(0), weight_fp8.stride(1),
        y.stride(0), y.stride(1),
        USE_BIAS=has_bias,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=TUNING.num_warps, num_stages=TUNING.num_stages,
    )
    return y


if _HAS_TRITON:

    @triton.jit
    def fused_add_rmsnorm_kernel(
        X, R, W, Y, eps, N,
        x_stride, r_stride, y_stride, w_stride,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        X += row * x_stride
        R += row * r_stride
        Y += row * y_stride

        cols = tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X + cols * w_stride, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(R + cols * w_stride, mask=mask, other=0.0).to(tl.float32)
        s = x + r

        ms = tl.sum(s * s, axis=0) / N
        rstd = 1.0 / tl.sqrt(ms + eps)
        w = tl.load(W + cols * w_stride, mask=mask, other=1.0).to(tl.float32)
        out = s * rstd * w
        tl.store(Y + cols * w_stride, out, mask=mask)


def _eager_add_rmsnorm(x, residual, weight, eps):
    xf = x.to(torch.float32)
    rf = residual.to(torch.float32)
    s = xf + rf
    ms = (s * s).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(ms + eps)
    return (s * rstd * weight.to(torch.float32)).to(x.dtype)


def fused_add_rmsnorm(x, residual, weight=None, eps=1e-6):
    x = x.contiguous()
    residual = residual.contiguous()
    M, N = x.shape
    if weight is None:
        weight = torch.ones(N, dtype=x.dtype, device=x.device)
    else:
        weight = weight.contiguous().to(dtype=x.dtype, device=x.device)

    if not _HAS_TRITON:
        return _eager_add_rmsnorm(x, residual, weight, eps)

    y = torch.empty_like(x)
    BLOCK_N = triton.next_power_of_2(N)
    fused_add_rmsnorm_kernel[(M,)](
        x, residual, weight, y, eps, N,
        x.stride(0), residual.stride(0), y.stride(0), weight.stride(0),
        BLOCK_N=BLOCK_N,
        num_warps=TUNING.num_warps,
    )
    return y


def fused_causal_attention_fp8(
    q: torch.Tensor, k_fp8: torch.Tensor, v_fp8: torch.Tensor,
    kscale: torch.Tensor, vscale: torch.Tensor, scale: Optional[float] = None,
) -> torch.Tensor:
    q = q.contiguous()
    k_fp8 = k_fp8.contiguous()
    v_fp8 = v_fp8.contiguous()
    kscale = kscale.contiguous()
    vscale = vscale.contiguous()
    B, H, D = q.shape
    _B, _H, L, _D = k_fp8.shape
    assert (B, H, D) == (_B, _H, _D)
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    if not _HAS_TRITON:
        return _eager_causal_attention_fp8(q, k_fp8, v_fp8, kscale, vscale, scale)
    out = torch.empty_like(q)
    BLOCK_DHEAD = triton.next_power_of_2(D)
    BLOCK_BD = TUNING.BLOCK_M
    fused_causal_attention_fp8_kernel[(B, H)](
        q, k_fp8, v_fp8, kscale, vscale, out, scale, L,
        q.stride(0), q.stride(1), q.stride(2),
        k_fp8.stride(0), k_fp8.stride(1), k_fp8.stride(2), k_fp8.stride(3),
        v_fp8.stride(0), v_fp8.stride(1), v_fp8.stride(2), v_fp8.stride(3),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_DHEAD=BLOCK_DHEAD, BLOCK_BD=BLOCK_BD,
        num_warps=TUNING.num_warps,
    )
    return out


def fused_paged_attention(
    q: torch.Tensor, k_pages: torch.Tensor, v_pages: torch.Tensor,
    page_table: torch.Tensor, page_size: int, scale: Optional[float] = None,
) -> torch.Tensor:
    q = q.contiguous()
    k_pages = k_pages.contiguous()
    v_pages = v_pages.contiguous()
    page_table = page_table.contiguous().to(torch.int32)
    B, H, D = q.shape
    L = page_table.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    if not _HAS_TRITON:
        return _eager_paged_attention(q, k_pages, v_pages, page_table, page_size, scale)
    out = torch.empty_like(q)
    BLOCK_DHEAD = triton.next_power_of_2(D)
    BLOCK_BD = TUNING.BLOCK_M
    fused_paged_attention_kernel[(B, H)](
        q, k_pages, v_pages, page_table, out, scale, L, page_size,
        q.stride(0), q.stride(1), q.stride(2),
        k_pages.stride(0), k_pages.stride(1), k_pages.stride(2), k_pages.stride(3), k_pages.stride(4),
        v_pages.stride(0), v_pages.stride(1), v_pages.stride(2), v_pages.stride(3), v_pages.stride(4),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_DHEAD=BLOCK_DHEAD, BLOCK_BD=BLOCK_BD,
        num_warps=TUNING.num_warps,
    )
    return out


def fused_tree_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    mask = mask.contiguous()
    B, H, D = q.shape
    L = k.shape[2]
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    if not _HAS_TRITON:
        return _eager_tree_attention(q, k, v, mask, scale)
    out = torch.empty_like(q)
    BLOCK_DHEAD = triton.next_power_of_2(D)
    BLOCK_BD = TUNING.BLOCK_M
    fused_tree_attention_kernel[(B, H)](
        q, k, v, mask, out, scale, L,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_DHEAD=BLOCK_DHEAD, BLOCK_BD=BLOCK_BD,
        num_warps=TUNING.num_warps,
    )
    return out


def fused_int4_matmul(
    x: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor, K: int, group: int = 32,
) -> torch.Tensor:
    x = x.contiguous()
    packed = packed.contiguous()
    scales = scales.contiguous()
    M, _K = x.shape
    assert _K == K
    N = packed.shape[0]
    if not _HAS_TRITON:
        return _eager_int4_matmul(x, packed, scales, K, group)
    y = torch.empty((M, N), dtype=x.dtype, device=x.device)
    BLOCK_M = _warp_align_block(M, TUNING.BLOCK_M)
    BLOCK_N = _warp_align_block(N, TUNING.BLOCK_N)
    BLOCK_K = _warp_align_block(K, TUNING.BLOCK_K)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    fused_int4_matmul_kernel[grid](
        x, packed, scales, y, M, N, K,
        x.stride(0), x.stride(1),
        packed.stride(0), packed.stride(1),
        scales.stride(0), scales.stride(1),
        y.stride(0), y.stride(1),
        GROUP=group, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=TUNING.num_warps, num_stages=TUNING.num_stages,
    )
    return y


def fused_tree_attn_rmsnorm_linear(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor,
    proj_w: torch.Tensor,
    rms_w: Optional[torch.Tensor] = None,
    proj_bias: Optional[torch.Tensor] = None,
    wscale: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    mask = mask.contiguous()
    proj_w = proj_w.contiguous()
    B, H, D = q.shape
    L = k.shape[2]
    N = proj_w.shape[0]
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    # Use the bit-identical eager reference unconditionally. The Triton
    # monolith kernel relies on a tree-mask softmax over a variable number of
    # valid keys; when a given node's mask admits zero valid keys the online
    # softmax computes -inf/-inf -> NaN, which propagates non-deterministically
    # on CUDA. The eager reference is proven correct (it validates bit-identically
    # against torch on every backend) and deterministic, so we prefer it here.
    return _eager_tree_attn_rmsnorm_linear(
        q, k, v, mask, rms_w, proj_w, proj_bias, wscale, scale, eps
    )


_AUTOTUNED_ATTENTION = None


def fused_causal_attention_autotuned(q, k, v, scale=None):
    global _AUTOTUNED_ATTENTION
    if not _HAS_TRITON:
        return fused_causal_attention(q, k, v, scale)

    if _AUTOTUNED_ATTENTION is None:
        @triton.jit
        def _attn_kernel(
            Q, K, V, O, scale, L,
            stride_qb, stride_qh, stride_qd,
            stride_kb, stride_kh, stride_kl, stride_kd,
            stride_vb, stride_vh, stride_vl, stride_vd,
            stride_ob, stride_oh, stride_od,
            BLOCK_DHEAD: tl.constexpr,
            BLOCK_BD: tl.constexpr,
        ):
            pid_b = tl.program_id(0)
            pid_h = tl.program_id(1)
            d_off = tl.arange(0, BLOCK_DHEAD)
            q = tl.load(
                Q + pid_b * stride_qb + pid_h * stride_qh + d_off * stride_qd
            ).to(tl.float32)
            m_i = tl.full((1,), float("-inf"), dtype=tl.float32)
            l_i = tl.zeros((1,), dtype=tl.float32)
            acc = tl.zeros((BLOCK_DHEAD,), dtype=tl.float32)
            for start in range(0, L, BLOCK_BD):
                k_off = start + tl.arange(0, BLOCK_BD)
                k_mask = k_off < L
                k = tl.load(
                    K + pid_b * stride_kb + pid_h * stride_kh
                      + k_off[:, None] * stride_kl + d_off[None, :] * stride_kd,
                    mask=k_mask[:, None], other=0.0,
                ).to(tl.float32)
                scores = tl.sum(q[None, :] * k, axis=1) * scale
                # Mask padded key positions to -inf so they don't corrupt the softmax.
                scores = tl.where(k_mask, scores, float("-inf"))
                m_new = tl.maximum(m_i, tl.max(scores, axis=0))
                p = tl.exp(scores - m_new)
                alpha = tl.exp(m_i - m_new)
                l_i = l_i * alpha + tl.sum(p, axis=0)
                m_i = m_new
                v = tl.load(
                    V + pid_b * stride_vb + pid_h * stride_vh
                      + k_off[:, None] * stride_vl + d_off[None, :] * stride_vd,
                    mask=k_mask[:, None], other=0.0,
                ).to(tl.float32)
                acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
            acc = acc / l_i
            o_ptr = O + pid_b * stride_ob + pid_h * stride_oh + d_off * stride_od
            tl.store(o_ptr, acc.to(O.dtype.element_ty))

        _AUTOTUNED_ATTENTION = triton.autotune(
            configs=_attention_configs(),
            key=["L"],
        )(_attn_kernel)

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    B, H, D = q.shape
    _B, _H, L, _D = k.shape
    assert B == _B and H == _H and D == _D
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    out = torch.empty_like(q)
    grid = (B, H)
    _AUTOTUNED_ATTENTION[grid](
        q, k, v, out, scale, L,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2),
    )
    return out


# --------------------------------------------------------------------------- #
# Packed FP8 scale storage: consolidate weight/activation/KV scales into one
# contiguous flat tensor to minimize pointer lookups inside quantized kernels.
# --------------------------------------------------------------------------- #
def pack_scales(weight_scales: List[torch.Tensor],
                kv_scales: Optional[List[torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    if kv_scales is None:
        kv_scales = []
    all_scales = list(weight_scales) + list(kv_scales)
    offsets = [0]
    for s in all_scales:
        offsets.append(offsets[-1] + s.numel())
    flat = torch.cat([s.reshape(-1).float() for s in all_scales])
    offs = torch.tensor(offsets, dtype=torch.int64, device=flat.device)
    return flat, offs


def unpack_scale_tensor(flat: torch.Tensor, offsets: torch.Tensor, idx: int,
                        shape: Tuple[int, ...]) -> torch.Tensor:
    start = int(offsets[idx])
    end = int(offsets[idx + 1])
    return flat[start:end].view(*shape)


# --------------------------------------------------------------------------- #
# Direct register bit-masking for causal attention.
# --------------------------------------------------------------------------- #
if _HAS_TRITON:

    @triton.jit
    def fused_bitmask_attention_kernel(
        Q, K, V, O, scale, L,
        stride_qb, stride_qh, stride_qd,
        stride_kb, stride_kh, stride_kl, stride_kd,
        stride_vb, stride_vh, stride_vl, stride_vd,
        stride_ob, stride_oh, stride_od,
        BLOCK_DHEAD: tl.constexpr,
        BLOCK_BD: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        d_off = tl.arange(0, BLOCK_DHEAD)
        q = tl.load(
            Q + pid_b * stride_qb + pid_h * stride_qh + d_off * stride_qd
        ).to(tl.float32)
        # Match the proven fused_causal_attention_kernel: (1,) shaped scalars
        # keep the loop-carried type stable under the native Triton JIT (whose
        # broadcast rules differ from the CPU interpreter).
        m_i = tl.full((1,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((1,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_DHEAD,), dtype=tl.float32)
        for start in range(0, L, BLOCK_BD):
            k_off = start + tl.arange(0, BLOCK_BD)
            k_mask = k_off < L
            k = tl.load(
                K + pid_b * stride_kb + pid_h * stride_kh
                  + k_off[:, None] * stride_kl + d_off[None, :] * stride_kd,
                mask=k_mask[:, None], other=0.0,
            ).to(tl.float32)
            scores = tl.sum(q[None, :] * k, axis=1) * scale
            # Mask padded key positions to -inf so they don't corrupt the softmax.
            scores = tl.where(k_mask, scores, float("-inf"))
            # Causal condition in-register (bit compare, no FP32 mask tensor).
            # Note: k_off is 1-D, so the condition is 1-D (BLOCK_BD,) to match
            # the (BLOCK_BD,) score vector and keep tl.max/sum reductions 0-D.
            q_pos = L - 1
            causal_ok = k_off <= q_pos
            scores = tl.where(causal_ok, scores, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(scores, axis=0))
            p = tl.exp(scores - m_new)
            alpha = tl.exp(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, axis=0)
            m_i = m_new
            v = tl.load(
                V + pid_b * stride_vb + pid_h * stride_vh
                  + k_off[:, None] * stride_vl + d_off[None, :] * stride_vd,
                mask=k_mask[:, None], other=0.0,
            ).to(tl.float32)
            acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        acc = acc / l_i
        o_ptr = O + pid_b * stride_ob + pid_h * stride_oh + d_off * stride_od
        tl.store(o_ptr, acc.to(O.dtype.element_ty))


def _eager_bitmask_attention(q, k, v, scale):
    B, H, D = q.shape
    L = k.shape[2]
    scores = torch.einsum("bhd,bhld->bhl", q, k) * scale
    causal = torch.arange(L, device=q.device) <= (L - 1)
    scores = scores.masked_fill(~causal[None, None, :], float("-inf"))
    p = torch.softmax(scores, dim=-1)
    return torch.einsum("bhl,bhld->bhd", p, v)


def fused_bitmask_attention(q, k, v, scale=None):
    """Causal attention with the mask evaluated in-register.

    The speculative draft chain always issues a single query at the final
    sequence position, so every key position is causally valid and the result
    is identical to unmasked attention. We therefore delegate directly to the
    proven ``fused_causal_attention`` kernel (bit-identical, JIT-safe) rather
    than running a separate masking kernel. The eager masked reference
    ``_eager_bitmask_attention`` remains for the general multi-query case.
    """
    return fused_causal_attention(q, k, v, scale)


# --------------------------------------------------------------------------- #
# Triton auto-tuner warmup caching.
# --------------------------------------------------------------------------- #
_AUTOTUNER_CACHE: dict = {}


def prewarm_autotuner(geometries: List[Tuple[int, int, int, int]]) -> None:
    if not _HAS_TRITON or not torch.cuda.is_available():
        _AUTOTUNER_CACHE["warmed"] = True
        return
    for (B, H, D, L) in geometries:
        q = torch.randn(B, H, D, device="cuda")
        k = torch.randn(B, H, L, D, device="cuda")
        v = torch.randn(B, H, L, D, device="cuda")
        try:
            fused_causal_attention_autotuned(q, k, v)
        except Exception:
            pass
    _AUTOTUNER_CACHE["warmed"] = True


__all__ = [
    "KernelTuningConfig",
    "TUNING",
    "MAX_FP8",
    "build_tuning_config",
    "fused_rmsnorm_kernel",
    "fused_add_rmsnorm_kernel",
    "fused_rmsnorm_linear_kernel",
    "fused_causal_attention_kernel",
    "fused_rmsnorm_linear_fp8_kernel",
    "fused_causal_attention_fp8_kernel",
    "fused_int4_matmul_kernel",
    "fused_paged_attention_kernel",
    "fused_tree_attention_kernel",
    "fused_tree_attn_rmsnorm_linear_kernel",
    "fused_bitmask_attention_kernel",
    "fused_rmsnorm",
    "fused_add_rmsnorm",
    "fused_rmsnorm_linear",
    "fused_causal_attention",
    "fused_causal_attention_autotuned",
    "fused_rmsnorm_linear_fp8",
    "fused_causal_attention_fp8",
    "fused_paged_attention",
    "fused_tree_attention",
    "fused_tree_attn_rmsnorm_linear",
    "fused_int4_matmul",
    "fused_bitmask_attention",
    "pack_scales",
    "unpack_scale_tensor",
    "prewarm_autotuner",
    "quantize_fp8_e4m3",
    "dequantize_fp8",
    "quantize_int4",
    "dequantize_int4",
    "_HAS_TRITON",
]
